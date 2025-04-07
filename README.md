# Einops Rearranger (Lark-based)

A lightweight numpy-based implementation of `einops.rearrange`, using `lark` for parsing the transformation patterns.

## Features
- Transpose, merge, split, repeat, and ellipsis-style reshaping
- Fully numpy compatible
- Custom pattern parser using Lark grammar (no regex hacks)
- Simple, composable API mimicking einops
- **No backpropagation** — this is not meant for training pipelines (The original one supports backpropagating through einops operations https://einops.rocks/2-einops-for-deep-learning/)
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

Design Decisions & How It Evolved

- Started with a Lark-based parser (Initial research into how to write parser led to this decision of using a parsing library as describing the grammar would be simpler and more elegant way for this than writing the logic from scratch, I.e instead of string splits or regex to handle complex nested patterns like (h w) cleanly and future-proof the parser logic)

- Defined a minimal custom grammar to parse einops-style patterns ("a b (c d) -> (a c) b d") with support for ..., groups, and regular axis names.

- Wrote a Lark Transformer that converts the parsed pattern into Python lists and tuples, so everything is structured before actual tensor logic kicks in.

- Implemented the rearrange() method step-by-step:

    - First version handled basic flattening and reshaping using axis name mapping.

    - Later added group splitting, group merging, and ellipsis support.

- Ellipsis support was added by detecting how many unspecified dims are left and mapping them under '...'.

- Added inference logic for cases where only one dimension in a group is unknown, e.g., in (h w), if h=3 and shape is 12, then w=4.

- Handled '1' as a special axis for inserting size-1 dims, useful for broadcasting/repeat-style logic.

- Added transpose step that only kicks in if input/output axis names match but in different orders.

- Cleanly separated logic the reshaping transposing operations into separate methods to make each step focused and testable.

- Added a debug flag, so internal print statements can be toggled easily without cluttering normal usage.
