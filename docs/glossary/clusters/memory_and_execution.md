# Cluster: Memory Hierarchy & Execution Boundaries

This is a **curated view** (Cluster) demonstrating the Zettelkasten approach to our Glossary architecture. It gathers specific terms related to hardware memory limits and compiler execution boundaries into a single thematic document.

By relying on the central Glossary database as the Single Source of Truth, we avoid duplicating definitions. Hover over the terms for a quick summary, or click them to dive deep into the original DB entry.

---

## 1. Execution Primitives
These terms dictate *how* and *where* execution happens when transitioning from Python to C++ and eventually to hardware loops.

- **[[glossary#ffi-foreign-function-interface|FFI (Foreign Function Interface)]]**
- **[[glossary#packedfunc|PackedFunc]]**
- **[[glossary#intrinsic-node|Intrinsic Node]]**
- **[[glossary#primfunc|PrimFunc]]**

## 2. Hardware Memory & Optimization
These terms describe the physical constraints of the hardware that necessitate complex compiler passes like Operator Fusion and Loop Tiling.

- **[[glossary#vram-global-memory|VRAM (Global Memory)]]**
- **[[glossary#l1-cache-shared-memory|L1 Cache (Shared Memory)]]**
- **[[glossary#operator-fusion|Operator Fusion]]**
- **[[glossary#loop-tiling|Loop Tiling]]**

--8<-- "docs/snippets/glossary_abbr.md"
