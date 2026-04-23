import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { AlertCircle, Info, Star } from "lucide-react";

interface ExplanationCardProps {
  method: string;
  imageUrl: string;
  onRate?: (ratings: Record<string, number>) => void;
}

const EVALUATION_CRITERIA = [
  {
    id: "covariate_complexity",
    label: "Covariate Complexity",
    description:
      "Are the features used in this explanation simple and easy to understand? (1 = very complex and hard to understand, 5 = very simple and easy to understand)",
  },
  {
    id: "compactness",
    label: "Compactness",
    description:
      "Is this explanation concise and free of unnecessary details? (1 = too long or overloaded, 5 = short and to the point)",
  },
  {
    id: "composition",
    label: "Composition",
    description:
      "Is the explanation clearly structured and well presented? (1 = poorly structured, 5 = very clear and well organized)",
  },
  {
    id: "context",
    label: "Context",
    description:
      "Does this explanation match your level of knowledge and needs? (1 = not relevant to me, 5 = highly relevant and appropriate)",
  },
  {
    id: "coherence",
    label: "Coherence",
    description:
      "Does the explanation align with logic and your expectations? (1 = confusing or contradictory, 5 = fully logical and consistent)",
  },
  {
    id: "controllability",
    label: "Controllability",
    description:
      "Can you interact with or adjust the explanation if needed? (1 = no control at all, 5 = full control and flexibility)",
  },
];

interface StarRatingProps {
  value: number;
  onChange: (value: number) => void;
}

function StarRating({ value, onChange }: StarRatingProps) {
  const [hoverValue, setHoverValue] = useState(0);

  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          onClick={() => onChange(star)}
          onMouseEnter={() => setHoverValue(star)}
          onMouseLeave={() => setHoverValue(0)}
          className="transition-transform hover:scale-110 active:scale-95"
        >
          <Star
            className={`w-6 h-6 transition-colors ${
              star <= (hoverValue || value)
                ? "fill-yellow-400 text-yellow-400"
                : "text-muted-foreground"
            }`}
          />
        </button>
      ))}
    </div>
  );
}

export function ExplanationCard({
  method,
  imageUrl,
  onRate,
}: ExplanationCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [ratings, setRatings] = useState<Record<string, number>>({});
  const [isZoomed, setIsZoomed] = useState(false);
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);

  const handleRatingChange = (criterionId: string, value: number) => {
    const newRatings = { ...ratings, [criterionId]: value };
    setRatings(newRatings);
    onRate?.(newRatings);
  };

  const allRated = EVALUATION_CRITERIA.every((criterion) =>
    ratings.hasOwnProperty(criterion.id)
  );

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative bg-card border border-border rounded-xl overflow-visible hover:border-purple-500/50 transition-all duration-300 shadow-lg"
      >
        <div className="p-5">
          <h3 className="mb-4 text-center">{method}</h3>

          {/* IMAGE */}
          <div
            className="relative cursor-pointer mb-4"
            onClick={() => setIsZoomed(true)}
          >
            <div className="relative rounded-lg overflow-hidden border border-border">
              <img
                src={imageUrl}
                alt={`${method} explanation`}
                className="w-full h-64 object-cover"
              />
            </div>
          </div>

          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full py-3 px-4 bg-secondary/50 hover:bg-secondary rounded-lg transition-colors flex items-center justify-between"
          >
            <span className="text-sm">
              {isExpanded ? "Hide" : "Show"} Evaluation Panel
            </span>
            {!allRated && !isExpanded && (
              <AlertCircle className="w-4 h-4 text-yellow-500" />
            )}
          </button>
        </div>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: "auto" }}
              exit={{ height: 0 }}
              className="overflow-visible border-t border-border"
            >
              <div className="p-5 space-y-6 bg-secondary/20 overflow-visible">
                {EVALUATION_CRITERIA.map((criterion) => (
                  <div key={criterion.id}>
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-2">
                        <label className="text-sm">
                          {criterion.label}
                        </label>

                        {/* TOOLTIP */}
                        <div className="relative group">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveTooltip(
                                activeTooltip === criterion.id
                                  ? null
                                  : criterion.id
                              );
                            }}
                            className="text-muted-foreground hover:text-foreground transition-colors"
                          >
                            <Info className="w-4 h-4" />
                          </button>

                          <div
                            className={`
                              absolute left-0 top-6
                              z-50
                              w-72
                              rounded-lg
                              border border-border
                              bg-popover
                              text-popover-foreground
                              shadow-xl
                              p-3
                              text-xs
                              transition-opacity
                              ${
                                activeTooltip === criterion.id
                                  ? "block"
                                  : "hidden"
                              }
                              md:group-hover:block
                            `}
                          >
                            {criterion.description}
                          </div>
                        </div>
                      </div>

                      <span
                        className={`text-sm px-2 py-0.5 rounded ${
                          ratings[criterion.id]
                            ? "bg-purple-500/20 text-purple-300"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {ratings[criterion.id] || "—"}
                      </span>
                    </div>

                    <StarRating
                      value={ratings[criterion.id] || 0}
                      onChange={(value) =>
                        handleRatingChange(criterion.id, value)
                      }
                    />
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* IMAGE ZOOM */}
      <AnimatePresence>
        {isZoomed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[9999] bg-black/90 flex items-center justify-center p-8"
            onClick={() => setIsZoomed(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="relative max-w-4xl w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <img
                src={imageUrl}
                alt={`${method} explanation zoomed`}
                className="w-full h-auto rounded-lg"
              />
              <button
                onClick={() => setIsZoomed(false)}
                className="absolute top-4 right-4 bg-black/70 hover:bg-black text-white rounded-full p-3"
              >
                ✕
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}


