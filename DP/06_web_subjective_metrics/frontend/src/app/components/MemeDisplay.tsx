import { motion } from "motion/react";

// interface MemeDisplayProps {
//   imageUrl: string;
// }

interface MemeDisplayProps {
  imageUrl: string;
  title: string;
  description: string;
  classification: string;
  // confidence: string;
}

export function MemeDisplay({
  imageUrl,
  title,
  description,
  classification,
  // confidence,
}: MemeDisplayProps) {
  return (
    <div className="mb-16">
      <h2 className="text-center mb-8">Current Meme</h2>

      <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-8 items-center">

        {/* Meme Image */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="relative"
        >
          <div className="rounded-xl overflow-hidden border-2 border-border shadow-2xl">
            <img
              src={imageUrl}
              alt="Meme to evaluate"
              className="w-full h-auto object-cover"
            />
          </div>
        </motion.div>

        {/* Meme Description */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-card/50 border border-border rounded-xl p-8 backdrop-blur-sm"
        >
          <h3 className="mb-4 text-purple-400">{title}</h3>

          <div className="space-y-4 text-muted-foreground">
            <p className="whitespace-pre-line">{description}</p>

            <div className="mt-6 pt-6 border-t border-border">
              <div className="flex items-center justify-between text-sm">
                <span className="text-foreground">AI Classification:</span>
                <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full">
                  {classification}
                </span>
              </div>

              {/* <div className="flex items-center justify-between text-sm mt-2">
                <span className="text-foreground">Confidence:</span>
                <span className="text-purple-300">{confidence}</span>
              </div> */}
            </div>
          </div>
        </motion.div>

      </div>
    </div>
  );
}
