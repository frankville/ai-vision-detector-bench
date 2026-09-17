# Model licensing

Why this document exists: the fastest model in a benchmark is worthless if you
cannot legally ship it. For a product deployed on customer premises, the licence
is a design constraint with the same weight as latency, and it is easier to
honour on day one than to unwind after the code is written.

This is an engineering summary, not legal advice. Confirm anything that affects
a commercial decision with a lawyer.

## The catalogue

| Model family | Licence | What it means in practice |
| --- | --- | --- |
| DETR (Meta) | Apache-2.0 | Use, modify and ship, including closed source. Keep the notice. |
| RT-DETR / RT-DETRv2 (Baidu) | Apache-2.0 | Same. |
| D-FINE | Apache-2.0 | Same. Code and COCO weights both. |
| RF-DETR Nano / Small / Base (Roboflow) | Apache-2.0 | Same. |
| RF-DETR Plus, XL, 2XL (Roboflow) | Roboflow PML 1.0 | Not in this catalogue. Separate terms, not a free commercial grant. |
| YOLOv5, v8, v11 (Ultralytics) | AGPL-3.0 | Copyleft that reaches across a network. See below. |
| YOLOX (Megvii) | Apache-2.0 | A YOLO-architecture baseline without the AGPL problem. |

## The AGPL question

AGPL-3.0 is GPL plus one clause that matters enormously for anything
server-side. Under plain GPL, obligations trigger on *distribution*. Under AGPL
section 13, they also trigger when users interact with the software **over a
network**. Running it only on your own servers is not a way around it.

For a video analytics product this means that if YOLO weights or the Ultralytics
package are part of a system users reach over a network, you owe those users the
complete corresponding source of that system, under AGPL, including your own
code that forms a single work with it. The scope of "single work" is genuinely
contested, which is itself the problem: a licence whose boundary you would have
to litigate is not a licence you want under a product.

The vendor sells a commercial licence that removes the obligation. That is a
legitimate route, and for some teams the right one. It is a purchasing decision
that should be made deliberately and early, not discovered during due diligence.

A detail that catches people out: the AGPL attaches to the Ultralytics *code*,
so it applies even when you train your own weights from scratch using their
tooling. Swapping the dataset does not swap the licence.

### Why the catalogue includes it anyway

YOLO is the reference every comparison is measured against, and omitting it
invites the question of whether it was omitted because it would have won. So it
is here, behind an optional extra, never installed by default, never in
`--all-permissive`, and flagged in yellow in every table.

Install it knowingly:

```bash
uv pip install -e '.[agpl-baseline]'
visionbench run -s clips/vtest.avi -m yolo11n -d 30
```

If it turns out to be meaningfully better than the permissive options on your
hardware, that is a real finding and worth knowing before you negotiate. If it
is not, you have evidence for choosing Apache-2.0 and the question is closed.

## Weights and datasets are separate questions

Three licences stack, and they are often different:

1. **The framework**, for example the `transformers` package, Apache-2.0.
2. **The weights**, which follow whoever trained them.
3. **The training data**. COCO images are Flickr photos under mixed terms, and
   the annotations are CC BY 4.0. This rarely constrains inference, but it can
   matter if you redistribute a derived dataset.

Fine-tuning a permissively licensed checkpoint on your own footage gives you
weights you own, subject to the base licence. Fine-tuning an AGPL one does not
launder the licence.

## Recommendation

Default to Apache-2.0 for anything that will ship. D-FINE and RT-DETRv2 are
strong real-time detectors under that licence, and RF-DETR's smaller variants
are too. Treat the AGPL entries as a measuring stick, and if one wins by enough
to matter, price the commercial licence before committing to it.
