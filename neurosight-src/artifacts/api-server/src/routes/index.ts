import { Router, type IRouter } from "express";
import healthRouter from "./health";
import analysisRouter from "./analysis";
import authRouter from "./auth";

const router: IRouter = Router();

router.use(healthRouter);
router.use(analysisRouter);
router.use(authRouter);

export default router;
