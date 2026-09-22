# SAM2-UNeXT architecture figure

## Figure contract

Purpose: describe the actual encoder, feature alignment, fusion and decoding computation in the supplied SAM2-UNeXT implementation. This is an architecture schematic, not evidence of performance or a proposed new method.

Archetype: schematic-led composite. Panel a explains the complete computation; panel b explains the trainable pre-block residual adapter; panel c explains the concatenation-before-upsampling decoder operation.

Export: 180 × 153 mm, white background, editable SVG text, embedded TrueType PDF text, 600 dpi PNG and LZW-compressed TIFF. Preview is 300 dpi. No specific target-journal upload requirements were supplied.

## Suggested English caption

**Fig. X | Architecture of SAM2-UNeXT.** **a,** An RGB image is processed by a SAM2 Hiera-L encoder with frozen backbone blocks and trainable adapters. A parallel frozen DINOv2 ViT-L/14 encoder receives the image resized to 448 × 448 pixels. Its final 1024-channel feature map is shared across four independent projection-and-resizing paths, each aligned to one Hiera stage. The paired features are concatenated and projected to 128 channels before progressive decoding. The decoder executes Up4, Up3, Up2 and Up1 in that order, followed by a 1 × 1 prediction convolution and bilinear resizing to produce full-resolution logits. **b,** A bottleneck adapter adds a learned residual to the input of each frozen Hiera block. **c,** Each skip-connected Up block concatenates its inputs before bilinear upsampling and two convolution–batch-normalization–ReLU sequences. Up4 has no skip input. C denotes channel concatenation; ch denotes channels. Spatial dimensions are nominal for compatible input sizes; decoder width scales in the same way as height.

## Source fidelity

- All stage/channel dimensions and operations come from the current implementation and its Hiera-L configuration.
- The diagram does not depict four independent DINO intermediate features.
- Backbone freezing and adapter training are distinguished. Main stage boxes contain both and therefore have solid outlines.
- The adapter is before each original Hiera block, not after it.
- The four fused features feed Up1, Up2, Up3, Up4 at their matching skip/input scales; actual decoder execution is bottom to top.
- Model output is logits. No sigmoid, ICCS, Mona, NightAdapter or proposed frequency module has been added.
- Padding used for shape alignment inside Up is omitted for clarity. Spatial dimensions assume compatible input sizes.
- The discarded first call to self.head(x) is not drawn as a second prediction head.
- The diagram is newly drawn from source, not a reused paper image. Cite the original SAM2-UNeXT work when using it as a baseline architecture in a manuscript.

## QA

- Source preflight: 21 PASS, 0 WARN, 0 FAIL.
- Final rendered panel alignment: PASS, 1.5 pt tolerance, no exemptions.
- Final PDF glyph-size audit: PASS; minimum 5.8 pt.
- Final PDF collision audit: PASS; 0 FAIL, 0 WARN. Text contained inside module fills is intentional.
- Visual review: checked all three panels and the assembled export, including decoder order, branch intersections, labels and arrow endpoints.
- Statistics/replicates: not applicable; no quantitative findings or predictions are depicted.
- No model inference was run. A source hash is retained in source_manifest.json.

## Reproduction

Run `python -B research/figures/sam2_unext/draw_architecture.py` from the project root. Python, Matplotlib and the installed project-local nature-figure alignment helper are used. The script directs its Matplotlib cache into this output folder. Model code is unchanged.
