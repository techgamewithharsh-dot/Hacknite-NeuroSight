# NeuroSight MRI Analyzer

Clinical workstation for comparing two MRI timepoints and reviewing progression vs pseudo-progression evidence.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/neurosight-analyzer/src/App.tsx` — primary analysis intake and results UI
- `artifacts/neurosight-analyzer/src/index.css` — visual tokens and workstation styling
- `artifacts/api-server/src/routes/analysis.ts` — preset/upload analysis response implementation
- `artifacts/api-server/src/routes/health.ts` — gateway and analyzer health endpoints
- `lib/api-spec/openapi.yaml` — source of truth for analysis and health contracts

## Architecture decisions

- The analysis contract mirrors the handoff specification: two scan inputs, three clinical metadata values, and the full structured result payload.
- Demo presets use the same result surface as uploaded scans so hackathon reviewers can reach the core experience without local NIfTI files.
- Uploads are sent as browser-native multipart form data; the generated shared client remains used for JSON preset requests and health queries.

## Product

NeuroSight lets neuro-oncology reviewers select a reference case or upload baseline/follow-up NIfTI scans, provide patient timing metadata, and inspect a verdict with confidence, slice previews, volume deltas, directional spread, and step-level inference telemetry.

## User preferences

- Preserve the existing input and output structure while redesigning the visual UI.

## Gotchas

- The shared generated library does not include browser Blob types by default; keep actual file uploads on the direct multipart path unless the shared library target is intentionally expanded.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
