# yoki5
Wraps h5py so reading and writing groups, datasets and attributes of complex HDF5 files is simpler.

## Role in ecosystem
Foundation storage utility on top of koyo.
Internal deps: koyo
Depended on by: annotine, imspy-ionglow, imzy, pyspa, qucee (imported, undeclared: imspy-core, imspy-data, imspy-multi, imspy-pre, imspy-quant, imspy-train)

## Key abstractions
- `base.Store` — main HDF5 file wrapper, subclassed by storage classes across the ecosystem
- `multi.MultiStore` — operate over several stores at once
- `attrs.Attributes` — attribute access helper
- `mixins` (`DisplayMixin`, `ColorMixin`) and `types` (`H5Protocol`, `H5MultiProtocol`)
- `export`, `repack`, `utilities` — export and repacking helpers

## Do not re-explain
When working in this repo, assume familiarity with: koyo
