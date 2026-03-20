import { Building2, Cross, Plane, MapPin, ChevronRight, Landmark } from "lucide-react";
import type { VenueDetection } from "@/data/venues";

const venueIcons = {
  stadium: Building2,
  hospital: Cross,
  airport: Plane,
  district: Landmark,
};

const VenueCard = ({
  venue,
  isActive,
  onClick,
}: {
  venue: VenueDetection;
  isActive: boolean;
  onClick: () => void;
}) => {
  const Icon = venueIcons[venue.type];

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-lg border transition-all duration-300 group ${
        isActive
          ? "border-primary/50 bg-primary/5 box-glow"
          : "border-border bg-card hover:border-primary/20 hover:bg-secondary/50"
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`p-2 rounded-md ${
              isActive ? "bg-primary/20 text-primary" : "bg-secondary text-muted-foreground"
            }`}
          >
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold text-sm text-foreground">{venue.name}</h3>
            <p className="text-xs text-muted-foreground capitalize mt-0.5">{venue.type}</p>
          </div>
        </div>
        <ChevronRight
          className={`w-4 h-4 mt-1 transition-transform ${
            isActive ? "text-primary rotate-90" : "text-muted-foreground"
          }`}
        />
      </div>

      <div className="mt-3 flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <MapPin className="w-3 h-3 text-primary" />
          <span className="text-xs font-mono text-primary">
            {venue.entrances.length} entrances
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-geo-success" />
          <span className="text-xs font-mono text-muted-foreground">
            {venue.confidence}% conf.
          </span>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="mt-3 h-1 bg-secondary rounded-full overflow-hidden">
        <div
          className="h-full bg-primary rounded-full transition-all duration-500"
          style={{ width: `${venue.confidence}%` }}
        />
      </div>
    </button>
  );
};

export default VenueCard;
