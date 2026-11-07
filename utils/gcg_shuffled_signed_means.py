"""Signed arithmetic means of raw saved score blocks; CPU/FP64 reduction."""
import numpy as np

METRICS = tuple(f'mu_{region}_{policy}' for policy in ('full', 'causal')
                for region in ('OO', 'UU', 'UO'))


def signed_means(scores, ranges):
    from utils.gcg_shuffled_common import region_layout
    blocks,masks,counts=region_layout(scores,ranges)
    values = {}
    for policy in ('full', 'causal'):
        for region, block in blocks.items():
            if not np.isfinite(block).all():
                raise ValueError('nonfinite raw score')
            mask = np.ones_like(masks[region]) if policy == 'full' else masks[region]
            selected = np.asarray(block, dtype=np.float64)[..., mask]
            values[f'mu_{region}_{policy}'] = selected.mean(axis=-1)
            counts[f'{region}_{policy}'] = int(mask.sum())
    np.testing.assert_array_equal(values['mu_UO_full'], values['mu_UO_causal'])
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError('nonfinite mean')
    return values, counts
