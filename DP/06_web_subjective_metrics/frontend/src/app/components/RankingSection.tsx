import { useState } from "react";
import { Method } from "../data/methods";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

interface RankingSectionProps {
  methods: Method[];
  onSubmit: (rankings: {
    best: Method;
    second: Method;
    least: Method;
  }) => void;
  allRated?: boolean;
}

export function RankingSection({
  methods,
  onSubmit,
  allRated = true,
}: RankingSectionProps) {
  const [rankings, setRankings] = useState<{
    best: Method | null;
    second: Method | null;
    least: Method | null;
  }>({
    best: null,
    second: null,
    least: null,
  });

  const [submitted, setSubmitted] = useState(false);

  const handleSelect = (
    position: "best" | "second" | "least",
    methodId: string
  ) => {
    const selected = methods.find((m) => m.id === methodId);
    if (!selected) return;

    const newRankings = { ...rankings };

    (Object.keys(newRankings) as Array<keyof typeof rankings>).forEach(
      (key) => {
        if (newRankings[key]?.id === selected.id) newRankings[key] = null;
      }
    );

    newRankings[position] = selected;
    setRankings(newRankings);
  };

  const handleSubmit = () => {
    if (!rankings.best || !rankings.second || !rankings.least) {
      alert("err: unranked");
      return;
    }

    if (submitted) return; 

    onSubmit({
      best: rankings.best,
      second: rankings.second,
      least: rankings.least,
    });

    setSubmitted(true);
    alert("Thank you for participating in the survey! :)");
  };

  const usedIds = Object.values(rankings)
    .filter((m): m is Method => m !== null)
    .map((m) => m.id);

  const availableMethods = (current: Method | null) =>
    methods.filter((m) => !usedIds.includes(m.id) || m.id === current?.id);

  return (
    <div className="mt-16 pt-16 border-t border-border">
      <h2 className="text-center mb-8">Final Ranking</h2>

      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <RankingSelect
          label="#1 Best Explanation"
          value={rankings.best}
          options={availableMethods(rankings.best)}
          onChange={(id) => handleSelect("best", id)}
        />

        <RankingSelect
          label="#2 Second Best"
          value={rankings.second}
          options={availableMethods(rankings.second)}
          onChange={(id) => handleSelect("second", id)}
        />

        <RankingSelect
          label="#3 Least Effective"
          value={rankings.least}
          options={availableMethods(rankings.least)}
          onChange={(id) => handleSelect("least", id)}
        />
      </div>

      <div className="mt-12 flex justify-center">
        <button
          onClick={handleSubmit}
          disabled={
            !rankings.best || !rankings.second || !rankings.least || !allRated || submitted
          }
          className={`px-8 py-4 rounded-lg transition-all duration-200 shadow-lg ${
            !rankings.best || !rankings.second || !rankings.least || !allRated || submitted
              ? "bg-secondary text-muted-foreground cursor-not-allowed opacity-50"
              : "bg-purple-600 hover:bg-purple-700"
          }`}
        >
          {submitted ? "Submitted" : "Submit Evaluation"}
        </button>
      </div>
    </div>
  );
}

function RankingSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: Method | null;
  options: Method[];
  onChange: (methodId: string) => void;
}) {
  return (
    <div className="border rounded-xl p-6 bg-secondary/20">
      <p className="mb-4 text-purple-400">{label}</p>

      <Select value={value?.id ?? ""} onValueChange={(val) => onChange(val)}>
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Select method" />
        </SelectTrigger>

        <SelectContent>
          {options.map((method) => (
            <SelectItem key={method.id} value={method.id}>
              {method.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}




