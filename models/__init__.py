# models/__init__.py
from models.unet    import UNet
from models.unetpp  import UNetPP
from models.deeplab import DeepLabV3Plus
from models.unetadv import UNetAdv, build_unetadv


def build_model(name: str, n_channels: int, n_classes: int, **kwargs):
    """
    Factory function — returns the right model by name.

    Args:
        name:       'unet' | 'unetpp' | 'deeplab' | 'unetadv'
        n_channels: number of input channels (4 for RGBA)
        n_classes:  number of output classes (1 for binary)
        **kwargs:   extra args passed to the model

    Returns:
        nn.Module
    """
    name = name.lower().strip()

    if name == 'unet':
        return UNet(n_channels=n_channels, n_classes=n_classes,
                    bilinear=kwargs.get('bilinear', False), dropout=kwargs.get('dropout', 0.2))

    elif name in ('unetpp', 'unet++'):
        return UNetPP(n_channels=n_channels, n_classes=n_classes,
                      deep_supervision=kwargs.get('deep_supervision', True),
                      dropout=kwargs.get('dropout', 0.2))

    elif name in ('deeplab', 'deeplabv3+', 'deeplabv3plus'):
        return DeepLabV3Plus(n_channels=n_channels, n_classes=n_classes)

    elif name in ('unetadv', 'va-unet++', 'variable-attention'):
        return UNetAdv(n_channels=n_channels, n_classes=n_classes,
                       deep_supervision=kwargs.get('deep_supervision', True),
                       dropout=kwargs.get('dropout', 0.5),
                       base_filters=kwargs.get('base_filters', [32, 64, 128, 256, 512]),
                       sk_reduction=kwargs.get('sk_reduction', 16))

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: unet, unetpp, deeplab, unetadv"
        )