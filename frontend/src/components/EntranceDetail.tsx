import type { EntranceMarker } from "@/data/venues";
import { MapPin, Shield, Truck, Star, ArrowRight } from "lucide-react";

const entranceIcons = {
  main: MapPin,
  emergency: Shield,
  service: Truck,
  vip: Star,
};

const entranceStyles = {
  main: "border-primary/30 bg-primary/5",
  emergency: "border-geo-danger/30 bg-geo-danger/5",
  service: "border-geo-amber/30 bg-geo-amber/5",
  vip: "border-geo-success/30 bg-geo-success/5",
};

const EntranceDetail = ({
  entrance,
  isSelected,
  onClick,
}: {
  entrance: EntranceMarker;
  isSelected: boolean;
  onClick?: () => void;
}) => {
  const Icon = entranceIcons[entrance.type];

  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border transition-all duration-300 cursor-pointer ${
        isSelected ? entranceStyles[entrance.type] + " box-glow" : "border-border bg-card hover:border-primary/20"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className={`w-4 h-4 ${isSelected ? "text-primary" : "text-muted-foreground"}`} />
          <span className="text-sm font-medium text-foreground">{entrance.label}</span>
        </div>
        <ArrowRight className="w-3 h-3 text-muted-foreground" />
      </div>
      <div className="mt-2 flex items-center gap-3">
        <span className="text-xs font-mono text-muted-foreground uppercase">
          {entrance.type}
        </span>
        <span className="text-xs font-mono text-primary">{entrance.confidence}%</span>
        <div className="flex-1 h-0.5 bg-secondary rounded-full overflow-hidden">
          <div
            className="h-full bg-primary rounded-full"
            style={{ width: `${entrance.confidence}%` }}
          />
        </div>
      </div>
    </button>
  );
};

export default EntranceDetail;
