import os
import json
import hashlib
import numpy as np
from PIL import Image

# 引入项目模块
from src.config import Config
from src.secret_sharing.moduli_gen import generate_secure_moduli
from src.secret_sharing.splitter import ImageCRTSplitter
from src.image_stego.dct_embed import DCTEmbedder

class AssetLocker:
    def __init__(self):
        self.embedder = DCTEmbedder()

    def lock_and_distribute(self, secret_img_path, pk_dir, cover_dir, output_dir, n, t):
        """
        执行完整的资产锁定流程
        """
        print("\n=== [Dealer] 启动资产锁定程序 ===")
        
        # 1. 收集公钥 (Identity Collection)
        print("[Step 1] 读取参与者公钥...")
        pk_files = sorted([f for f in os.listdir(pk_dir) if f.endswith('.pk')])
        available_pk = len(pk_files)
        
        if available_pk < n:
            raise ValueError(f"参与者不足! (公钥数 {available_pk} < 需要 {n})")
        
        if n < t:
            raise ValueError(f"份额数量不足! (份额数 {n} < 门限 {t})")
            
        # 只使用前n个公钥
        pk_files = pk_files[:n]
        public_keys = []
        for pk_f in pk_files:
            with open(os.path.join(pk_dir, pk_f), 'r') as f:
                pk_data = json.load(f)
                pk_data['_filename'] = pk_f # 暂存文件名用于标记
                public_keys.append(pk_data)
        print(f"   -> 已加载 {n} 个数字身份")

        # 2. 动态参数生成 (Math Setup)
        print("[Step 2] 生成抗量子与CRT参数...")
        moduli = generate_secure_moduli(n, t)
        
        # 3. 资产分割 (Splitting)
        print(f"[Step 3] 切割秘密图像: {os.path.basename(secret_img_path)}")
        img = Image.open(secret_img_path).convert('RGB') # 确保 RGB
        img_arr = np.array(img)
        
        splitter = ImageCRTSplitter(n, t, moduli)
        shares = splitter.split(img_arr) # 返回 SharePayload 列表

        # 4. 锚定与分发 (Anchoring & Distribution)
        print("[Step 4] 锚定权益并嵌入载体...")
        manifest_registry = []
        
        # 准备载体图
        cover_files = sorted([os.path.join(cover_dir, f) for f in os.listdir(cover_dir) 
                             if f.lower().endswith(('.png', '.jpg'))])
        if len(cover_files) < n:
            raise ValueError(f"载体图像不足! 需要 {n} 张")

        # 确保输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(n):
            share = shares[i]
            target_pk = public_keys[i]
            cover_path = cover_files[i]
            
            # --- A. 序列化与指纹 ---
            share_bytes = share.to_bytes()
            # 计算影子数据的哈希 (这是未来验证的唯一凭证)
            share_hash = hashlib.sha256(share_bytes).hexdigest()
            
            # 计算公钥指纹 (简单 Hash 用于索引)
            pk_json = json.dumps(target_pk['t'], sort_keys=True).encode()
            pk_fingerprint = hashlib.sha256(pk_json).hexdigest()
            
            # --- B. 隐写嵌入 ---
            # 真正的"藏"过程
            print(f"   -> 正在处理第 {i+1} 份 (归属: {target_pk['_filename']})...")
            stego_img = self.embedder.embed(cover_path, share_bytes)
            
            # 保存结果
            out_filename = f"locked_asset_{i+1}.png"
            out_path = os.path.join(output_dir, out_filename)
            stego_img.save(out_path)
            
            # --- C. 记录清单 ---
            # 这里的每一条记录都是一份"所有权声明"
            entry = {
                "share_index": i,
                "modulus": moduli[i],
                "carrier_file": out_filename,
                "share_fingerprint": share_hash,      # 锁：数据的哈希
                "owner_pk_fingerprint": pk_fingerprint, # 钥匙孔：公钥的哈希
                "owner_alias": target_pk['_filename']   # 备注
            }
            manifest_registry.append(entry)

        # 5. 发布资产清单 (Manifest)
        print("[Step 5] 签署并发布资产清单...")
        manifest = {
            "version": "QSP-2.0",
            "threshold": t,
            "total_shares": n,
            "public_seed": public_keys[0]['public_seed'], # 记录用于矩阵 A 的种子
            "registry": manifest_registry
        }
        
        manifest_path = os.path.join(output_dir, "asset_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
            
        print("\n✅ 资产锁定完成!")
        print(f"📂 分发目录: {output_dir}")
        print(f"📜 资产清单: asset_manifest.json")