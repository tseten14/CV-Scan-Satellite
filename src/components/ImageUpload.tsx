import { useCallback, useState } from "react";
import { Upload, Image as ImageIcon, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  isProcessing: boolean;
  hasPin: boolean;
}

const ImageUpload = ({ onImageSelect, isProcessing, hasPin }: ImageUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onImageSelect(file);
    },
    [onImageSelect]
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex h-full flex-col items-center justify-center p-8"
    >
      {!hasPin ? (
        <div className="text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-border bg-secondary">
            <ImageIcon className="h-7 w-7 text-muted-foreground" />
          </div>
          <p className="font-mono text-sm text-muted-foreground">
            Select a building on the map first
          </p>
        </div>
      ) : (
        <label
          className={`group relative flex w-full max-w-md cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-12 transition-all ${
            isDragging
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-secondary/50"
          } ${isProcessing ? "pointer-events-none opacity-50" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileInput}
            disabled={isProcessing}
          />

          <AnimatePresence mode="wait">
            {isProcessing ? (
              <motion.div
                key="processing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center gap-4"
              >
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
                <div className="text-center">
                  <p className="font-mono text-sm font-semibold text-primary">
                    Running inference pipeline...
                  </p>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">
                    YOLOv8 · Door Detection Model
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="upload"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center gap-4"
              >
                <div className="flex h-14 w-14 items-center justify-center rounded-full border border-primary/30 bg-primary/10 transition-colors group-hover:bg-primary/20">
                  <Upload className="h-6 w-6 text-primary" />
                </div>
                <div className="text-center">
                  <p className="font-mono text-sm font-semibold text-foreground">
                    Upload facade image
                  </p>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">
                    Drag & drop or click · PNG, JPG up to 20MB
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </label>
      )}
    </motion.div>
  );
};

export default ImageUpload;
