"""Signed arithmetic means of raw saved score blocks; CPU/FP64 reduction."""
import numpy as np

METRICS = tuple(f'mu_{region}_{policy}' for policy in ('full', 'causal')
                for region in ('OO', 'UU', 'UO'))


def signed_means(scores, ranges):
    if scores.ndim < 2 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError('square score matrices required')
    a, b = ranges['original']; c, d = ranges['suffix']
    if not 0 <= a < b <= c < d <= scores.shape[-1] or d-c != 20:
        raise ValueError('invalid Original/Suffix ranges or suffix length')
    o = np.arange(a, b); u = np.arange(c, d)
    blocks = {'OO': scores[..., a:b, a:b], 'UU': scores[..., c:d, c:d],
              'UO': scores[..., c:d, a:b]}
    masks = {'OO': o[None, :] <= o[:, None], 'UU': u[None, :] <= u[:, None],
             'UO': o[None, :] <= u[:, None]}
    values = {}; counts = {'original_length': b-a, 'suffix_length': d-c}
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
