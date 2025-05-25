def count_parameters(model, unit='M', fmt='.2f'):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    # 单位换算
    divisor = {
        None: 1,
        'K': 1e3,
        'M': 1e6
    }.get(unit, 1e6)
    
    if unit:
        trainable_str = f"{trainable/divisor:{fmt}}{unit}"
        total_str = f"{total/divisor:{fmt}}{unit}"
        return (trainable_str, total_str)
    return (trainable, total)