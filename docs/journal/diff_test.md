# Diff Rendering Test

This page tests the experimental `diff2html` side-by-side diff integration.

<div class="diff-code" style="display: none;">
--- a/example.py
+++ b/example.py
@@ -1,4 +1,5 @@
 def hello_world():
-    print("Hello")
-    return False
+    print("Hello, World!")
+    print("This is a side-by-side diff test.")
+    return True
</div>
