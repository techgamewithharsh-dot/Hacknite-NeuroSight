import { Router, type IRouter } from "express";
import { AnalyzerHealthResponse, HealthCheckResponse } from "@workspace/api-zod";

const router: IRouter = Router();

router.get("/healthz", (_req, res) => {
  const data = HealthCheckResponse.parse({ status: "ok" });
  res.json(data);
});

router.get("/v1/health", (_req, res) => {
  const data = AnalyzerHealthResponse.parse({
    status: "online",
    service: "Brain Tumor Progression Analyzer API",
    triton_connected: true,
    serving_backend: "NVIDIA Triton Inference Server",
    timestamp: new Date().toISOString(),
  });
  res.json(data);
});

export default router;
