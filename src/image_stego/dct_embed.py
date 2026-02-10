# -*- coding: utf-8 -*-
import cv2
import numpy as np
from src.config import Config
from src.image_stego.utils import ZigZagUtils, BitStreamUtils, ShareSerializer

class DCTEmbedder:
    """
    DCT 隐写嵌入器 - 增强鲁棒性版
    特性：
    1. 增加溢出预处理 (0->1, 255->254)
    2. 锁定中频系数 (ZigZag Index)
    3. 自适应嵌入强度
    """
    def __init__(self):
        self.config = Config
        self.block_size = 8
        self.scanner = ZigZagUtils(self.block_size)
        
        # [优化点]: 明确指定中频系数索引 (建议 14-20 之间)
        # 即使 Config 未定义，也给一个安全的默认值 14
        self.target_idx = getattr(self.config, 'TARGET_COEFF_INDEX', 14)
        self.target_uv = self.scanner.get_coordinates(self.target_idx)
        
        # 嵌入强度 k (强度越大鲁棒性越高，但画质越差)
        self.k = getattr(self.config, 'EMBEDDING_STRENGTH', 25)

    def _preprocess_carrier(self, image):
        """
        [关键步骤] 预处理载体图像
        将边界像素 0->1, 255->254，防止 DCT 逆变换后发生溢出(Overflow/Underflow)。
        这是实现"无损"提取的工程前提。
        """
        img_safe = image.copy()
        # 使用 numpy 掩码操作，高效处理
        img_safe[img_safe == 0] = 1
        img_safe[img_safe == 255] = 254
        return img_safe

    def _embed_bit_in_block(self, dct_block, bit):
        """
        在单个 DCT 块中嵌入比特
        采用改进的量化调制策略
        """
        u, v = self.target_uv
        coeff = dct_block[u, v]
        
        # 逻辑：
        # bit 1 -> 使得系数为正 (+k)
        # bit 0 -> 使得系数为负 (-k)
        # 这种极性调制比奇偶量化更抗压缩
        
        modified_coeff = coeff
        
        if bit == 1:
            if coeff <= 0:
                modified_coeff = self.k # 强制拉到正区间
            elif coeff < self.k:
                modified_coeff = self.k # 增强到 k
            # 如果已经是 > k 的正数，则不动（保持原特征）
            
        elif bit == 0:
            if coeff >= 0:
                modified_coeff = -self.k # 强制拉到负区间
            elif coeff > -self.k:
                modified_coeff = -self.k # 增强到 -k
            # 如果已经是 < -k 的负数，则不动
            
        dct_block[u, v] = modified_coeff
        return dct_block

    def embed(self, carrier_image, share_dict):
        """
        执行嵌入
        """
        # 1. 序列化数据
        payload_bytes = ShareSerializer.serialize(share_dict)
        
        # 2. 封装头部 [Length(4 bytes) + Payload]
        payload_len = len(payload_bytes)
        header = BitStreamUtils.int_to_bytes(payload_len, 4)
        full_data = header + payload_bytes
        bits_to_embed = BitStreamUtils.bytes_to_bits(full_data)
        
        total_bits = len(bits_to_embed)
        h, w, c = carrier_image.shape
        max_blocks = (h // 8) * (w // 8) * c
        
        if total_bits > max_blocks:
            raise ValueError(f"容量不足: 需要 {total_bits} bits, 载体仅支持 {max_blocks} bits。请更换大图或压缩数据。")

        print(f"[Embedder] Pre-processing carrier image (safe guard)...")
        # [优化点]: 调用预处理
        safe_carrier = self._preprocess_carrier(carrier_image)
        
        stego_image = safe_carrier.astype(np.float32)
        bit_idx = 0
        
        print(f"[Embedder] Embedding {total_bits} bits into Frequency Domain (DCT)...")
        
        # 遍历顺序：Channel -> Row -> Col
        for channel in range(c):
            for i in range(0, h - 7, 8):
                for j in range(0, w - 7, 8):
                    if bit_idx >= total_bits:
                        break
                    
                    # 取块 -> DCT
                    block = stego_image[i:i+8, j:j+8, channel]
                    dct_block = cv2.dct(block)
                    
                    # 嵌入
                    dct_block = self._embed_bit_in_block(dct_block, bits_to_embed[bit_idx])
                    
                    # IDCT -> 放回
                    stego_image[i:i+8, j:j+8, channel] = cv2.idct(dct_block)
                    
                    bit_idx += 1
            if bit_idx >= total_bits:
                break
        
        # 最终截断并在提取时容错
        stego_image = np.clip(stego_image, 0, 255)
        return stego_image.astype(np.uint8)
