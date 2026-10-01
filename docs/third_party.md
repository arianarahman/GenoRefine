# Third-party software and source pins

Third-party components remain governed by their own terms. A commit or checksum establishes source identity; it does not replace upstream licensing, citation, or attribution requirements.

## Official comparator pins

| Component | Official repository | Frozen version/commit | How the pin is used |
|---|---|---|---|
| IDEC | <https://github.com/XifengGuo/IDEC> | commit `4a661c211f082022249758ab3312f7db21e2aa99` | Package 1 validates a clean official checkout and preserves the documented architecture, assignment, target-distribution, and objective behavior while applying API compatibility updates. |
| SpaGCN | <https://github.com/jianhuupenn/SpaGCN> | version 1.2.7, commit `dc7a1c26ea0fdf4dfe7064adc7699be141b4871f` | Package 4 validates hashes of selected official source files and uses a device-compatible port with explicitly bounded changes. The upstream MIT notice is preserved in `THIRD_PARTY_NOTICES.md` and `third_party/licenses/SpaGCN-LICENSE`. |
| GraphST | <https://github.com/JinmiaoChenLab/GraphST> | version 1.1.1, commit `d62b0b7b6cd38ee285f3ac8cd67b7341a10bcc74` | Package 4b downloads a commit-addressed archive, verifies its SHA-256 and size, and verifies each selected source file. The upstream AGPL-3.0 text is retained at `third_party/licenses/GraphST-LICENSE.md`. |
| PASTE | <https://github.com/raphael-group/paste> | version 1.4.0, commit `a9b10b24ba33e94a89dd89e8ee5e4900e18b1886` | Package 4b uses the same archive/member verification for label-free donor-pair alignment. The upstream BSD 3-Clause text is retained at `third_party/licenses/PASTE-LICENSE`. |

The authoritative fields are in `independent_idec_panel_v2.json`, `spatial_multisection_panel_v1.json`, and `spatial_graphst_panel_v1.json`. GraphST/PASTE archive acquisition is implemented by `revision_pipeline/spatial_graphst/source_acquisition.py`. Official source checkouts and fetched archives are excluded from version control. The frozen v1 container recipes remain byte-identical to their audited hashes; the repository stores the upstream notices separately, and redistributed images must retain the applicable notices. The Ariana Rahman author header in `spagcn_compat.py` identifies authorship of the local compatibility adaptation; it does not replace SpaGCN's upstream copyright.

GraphST's pinned root `LICENSE.md` is AGPL-3.0, although the same commit's
`setup.py` metadata reports MIT. This release preserves the root license text
and treats AGPL-3.0 as the applicable upstream notice unless the GraphST
maintainers clarify the metadata.

## Bundled GenoMap snapshot

`revision_pipeline/vendor/` includes a byte-for-byte snapshot of GenoMap 1.3.6 files. `revision_pipeline/vendor/source_manifest.json` identifies every bundled file, size, and SHA-256 digest and records that no modifications were made.

The upstream license text is included at:

`revision_pipeline/vendor/genomap-1.3.6.dist-info/LICENSE.txt`

That file identifies the terms as **Creative Commons Attribution-NonCommercial-NoDerivs 2.0**. Those terms apply to the identified GenoMap snapshot only. They do not provide a license for GenoRefine, and compatibility with the future repository-wide license must be reviewed before release.

## Dependency environments

Pinned Python package inventories and container recipes are stored under `revision_pipeline/environment/`, `revision_pipeline/spatial_multisection/`, and `revision_pipeline/spatial_graphst/`. Each dependency retains its upstream license. Before redistribution, generate or review a complete software bill of materials and preserve notices required by every dependency and base image.
