import { useId } from "react";

/**
 * Site wordmark: dual orbit + core — reads at favicon and header sizes, matches cyan→violet gradient.
 */
export function GeoAiMark({ className = "h-10 w-10" }: { className?: string }) {
  const uid = useId().replace(/:/g, "");
  const gradId = `geoai-grad-${uid}`;

  return (
    <svg
      className={className}
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <defs>
        <linearGradient id={gradId} x1="6" y1="4" x2="36" y2="38" gradientUnits="userSpaceOnUse">
          <stop stopColor="#38bdf8" />
          <stop offset="0.45" stopColor="#22d3ee" />
          <stop offset="1" stopColor="#a78bfa" />
        </linearGradient>
      </defs>
      {/* Crossed orbits (satellite / geospatial) */}
      <ellipse
        cx="20"
        cy="20"
        rx="16"
        ry="7"
        transform="rotate(-28 20 20)"
        stroke={`url(#${gradId})`}
        strokeWidth="1.35"
        strokeLinecap="round"
        opacity={0.9}
      />
      <ellipse
        cx="20"
        cy="20"
        rx="16"
        ry="7"
        transform="rotate(62 20 20)"
        stroke={`url(#${gradId})`}
        strokeWidth="1.35"
        strokeLinecap="round"
        opacity={0.9}
      />
      {/* Ground / scan plane */}
      <path
        d="M6 28 Q20 24 34 28"
        stroke={`url(#${gradId})`}
        strokeWidth="1.2"
        strokeLinecap="round"
        opacity={0.55}
      />
      {/* Core */}
      <circle cx="20" cy="20" r="6.25" fill={`url(#${gradId})`} opacity={0.95} />
      <circle cx="20" cy="20" r="3.2" fill="hsl(222, 28%, 5%)" />
      <circle cx="20" cy="20" r="1.35" fill={`url(#${gradId})`} opacity={0.9} />
    </svg>
  );
}
