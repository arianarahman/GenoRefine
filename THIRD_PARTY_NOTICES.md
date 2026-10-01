# Third-party notices

This file preserves notices required for third-party source that is included in
or substantially represented by the GenoRefine release. Additional dependency
and source-pin information is available in [`docs/third_party.md`](docs/third_party.md).

## GraphST and PASTE comparator sources

The Package 4b acquisition workflow retrieves selected source files from the
official GraphST 1.1.1 and PASTE 1.4.0 repositories at frozen commits. Their
upstream license texts are preserved here:

- [GraphST GNU Affero General Public License, version 3](third_party/licenses/GraphST-LICENSE.md)
- [PASTE BSD 3-Clause License](third_party/licenses/PASTE-LICENSE)

These licenses govern the respective upstream components, not the entire
GenoRefine repository. The frozen v1 container recipe remains byte-identical
to the audited execution recipe; anyone redistributing a rebuilt image must
carry the applicable upstream notices with that image.

GraphST's pinned root `LICENSE.md` contains AGPL-3.0, while its `setup.py`
metadata says MIT. This release preserves and follows the root license text for
the pinned source and does not characterize GraphST as MIT.

## SpaGCN compatibility port

`revision_pipeline/spatial_multisection/spagcn_compat.py` is a
device-compatible adaptation that closely follows SpaGCN 1.2.7
`simple_GC_DEC.fit` from commit
`dc7a1c26ea0fdf4dfe7064adc7699be141b4871f`. The upstream notice is reproduced
below and preserved separately at
[`third_party/licenses/SpaGCN-LICENSE`](third_party/licenses/SpaGCN-LICENSE).

> MIT License
>
> Copyright (c) 2020 JianHu
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Bundled GenoMap snapshot

The byte-for-byte GenoMap 1.3.6 snapshot under `revision_pipeline/vendor/`
retains its upstream license at
`revision_pipeline/vendor/genomap-1.3.6.dist-info/LICENSE.txt`. See
[`docs/third_party.md`](docs/third_party.md) for its source manifest and scope.
