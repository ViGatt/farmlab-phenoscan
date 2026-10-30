# Arquitetura U-Net para segmentação binária de anomalias NDVI

# Entrada : (B, 2, 40, 54)  — canal 0: NDVI atual | canal 1: desvio do baseline
# Saída   : (B, 1, 40, 54)  — máscara de anomalia (sigmoid → [0,1])

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlocoConv(nn.Module):
    """Dois conv 3×3 com BatchNorm e ReLU."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.bloco = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bloco(x)


class UNet(nn.Module):
    """
    U-Net de 3 níveis adaptado para imagens pequenas (40×54).
    Canais: 2 → 16 → 32 → 64 (bottleneck) → 32 → 16 → 1
    """
    def __init__(self, in_channels: int = 2, base_ch: int = 16):
        super().__init__()

        # Encoder
        self.enc1 = BlocoConv(in_channels, base_ch)       # (B, 16, 40, 56)
        self.enc2 = BlocoConv(base_ch,     base_ch * 2)   # (B, 32, 20, 28)
        self.enc3 = BlocoConv(base_ch * 2, base_ch * 4)   # (B, 64, 10, 14)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = BlocoConv(base_ch * 4, base_ch * 8)  # (B, 128, 5, 7)

        # Decoder
        self.up3   = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3  = BlocoConv(base_ch * 8, base_ch * 4)  # skip de enc3

        self.up2   = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2  = BlocoConv(base_ch * 4, base_ch * 2)  # skip de enc2

        self.up1   = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1  = BlocoConv(base_ch * 2, base_ch)      # skip de enc1

        # Saída
        self.saida = nn.Conv2d(base_ch, 1, 1)

    def _pad(self, x):
        """Padding reflect: (40, 54) → (40, 56)."""
        return F.pad(x, (1, 1, 0, 0), mode="reflect")  # pad esquerda/direita

    def _crop(self, x):
        """Remove o padding: (40, 56) → (40, 54)."""
        return x[:, :, :, 1:-1]

    def forward(self, x):
        # Sem padding interno — o Dataset garante tamanho correto
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b  = self.bottleneck(self.pool(s3))
        d3 = self.dec3(torch.cat([self.up3(b),  s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.saida(d1)

# ── Funções de perda ─────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, alvo: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice Loss para segmentação binária com classes desbalanceadas."""
    pred  = torch.sigmoid(pred).view(-1)
    alvo  = alvo.float().view(-1)
    inter = (pred * alvo).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + alvo.sum() + eps)


def perda_combinada(pred: torch.Tensor, alvo: torch.Tensor,
                    peso_dice: float = 0.5) -> torch.Tensor:
    """BCE + Dice Loss (50/50 por padrão)."""
    bce  = F.binary_cross_entropy_with_logits(pred, alvo.float())
    dice = dice_loss(pred, alvo)
    return peso_dice * dice + (1 - peso_dice) * bce


# ── Teste rápido de forma ────────────────────────────────────────────────────

if __name__ == "__main__":
    modelo = UNet(in_channels=2)
    x = torch.randn(4, 2, 40, 54)   # batch de 4 imagens
    y = modelo(x)
    print(f"Entrada : {x.shape}")
    print(f"Saída   : {y.shape}")
    assert y.shape == (4, 1, 40, 54), "Shape incorreto!"
    print("Arquitetura OK.")

    # Conta parâmetros
    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis: {n_params:,}")