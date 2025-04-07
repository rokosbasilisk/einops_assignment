# Einops Rearranger (Lark-based)

A lightweight numpy-based implementation of `einops.rearrange`, using `lark` for parsing the transformation patterns.

## Features
- Transpose, merge, split, repeat, and ellipsis-style reshaping
- Fully numpy compatible
- Custom pattern parser using Lark grammar (no regex hacks)
- Simple, composable API mimicking einops
- **No backpropagation** — this is not meant for training pipelines
- **Not built on PyTorch** — this is pure numpy, so use only for static reshaping tasks

## Usage

```python
rearranger = EinopsRearranger()

x = np.random.rand(3, 4)
print("Transpose:", rearranger.rearrange(x, 'h w -> w h').shape)  # (4, 3)

x = np.random.rand(3, 4, 5)
print("Merge:", rearranger.rearrange(x, 'a b c -> (a b) c').shape)  # (12, 5)

x = np.random.rand(12, 10)
print("Split:", rearranger.rearrange(x, '(h w) c -> h w c', h=3).shape)  # (3, 4, 10)

x = np.random.rand(3, 1, 5)
print("Repeat:", rearranger.rearrange(x, 'a 1 c -> a b c', b=4).shape)  # (3, 4, 5)

x = np.random.rand(2, 3, 4, 5)
print("Ellipsis:", rearranger.rearrange(x, '... h w -> ... (h w)').shape)  # (2, 3, 20)
```

Design Decisions

- Used Lark parser instead of string manipulation to support nested and grouped axes cleanly
- Retained group parsing like (h w) as tuples internally for flexible manipulation
- Flattening logic is recursive and handles nested groups properly
- Ellipsis is expanded by calculating missing dimensions and assigning placeholder axis names
- Explicit axis lengths (like h=3) are passed via kwargs and auto-resolve shape inference
- Pattern errors (missing axes or mismatch in dimensions) raise clear exceptions
- Focused only on static numpy reshaping — no support for backprop currently

