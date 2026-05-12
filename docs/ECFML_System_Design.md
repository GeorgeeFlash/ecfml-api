# ECFML — System Design Document
**Electricity Consumption Forecasting using Machine Learning**
*A Case Study of the North West Region of Cameroon*

| Field | Value |
|---|---|
| Author | Sanda Elvis Toge (UBa22S1297) |
| Supervisor | Prof. Fautso Gaetan |
| Version | 1.0 |
| Date | May 2026 |
| Stack | Next.js 16 · FastAPI · Prisma 7 · Clerk v7 · Recharts · Vercel AI SDK v6 · Neon Postgres · Vercel · Render |

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [System Architecture](#3-system-architecture)
4. [Component Breakdown](#4-component-breakdown)
5. [Database Schema (Prisma 7)](#5-database-schema-prisma-7)
6. [API Design (FastAPI)](#6-api-design-fastapi)
7. [ML Pipeline Design](#7-ml-pipeline-design)
8. [UI/UX Wireframes & User Flows](#8-uiux-wireframes--user-flows)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Environment Variables](#11-environment-variables)
12. [Error Handling Strategy](#12-error-handling-strategy)

---

## 1. System Overview

ECFML is a two-service web application:

- **Frontend** — Next.js 16 (App Router) deployed on Vercel. Handles authentication, file upload, dashboard UI, chart rendering, and proxies all ML-related requests to the FastAPI service.
- **ML Backend** — FastAPI (Python 3.12) deployed on Render. Handles all data processing, model training, evaluation, and forecasting. CPU-bound training runs as background tasks.

They share a **Neon Postgres** database accessed by the frontend via **Prisma 7**.

```
Browser
  │
  ▼
Next.js 16 (Vercel)           ←──── Clerk v7 (Auth)
  │   │                              Uploadthing (File Storage)
  │   └── /api/* routes (proxy) ──► FastAPI (Render)
  │                                       │
  ▼                                       ▼
Prisma 7 ──────────────────────► Neon Postgres
```

---

## 2. Repository Structure

Two separate Git repositories (or two root-level folders in one repo with independent `package.json` / `requirements.txt`). No monorepo tooling required.

### 2.1 Frontend — `ecfml-web/`

```
ecfml-web/
├── src/
│   ├── app/                          # Next.js 16 App Router
│   │   ├── layout.tsx                # Root layout (Clerk provider, fonts)
│   │   ├── page.tsx                  # Landing / redirect to dashboard
│   │   ├── (auth)/                   # Route group — public auth pages
│   │   │   ├── sign-in/
│   │   │   │   └── [[...sign-in]]/
│   │   │   │       └── page.tsx      # Clerk <SignIn /> component
│   │   │   └── sign-up/
│   │   │       └── [[...sign-up]]/
│   │   │           └── page.tsx      # Clerk <SignUp /> component
│   │   ├── (dashboard)/              # Route group — protected pages
│   │   │   ├── layout.tsx            # Dashboard shell (sidebar, header)
│   │   │   ├── dashboard/
│   │   │   │   └── page.tsx          # Overview cards
│   │   │   ├── dashboard/data/
│   │   │   │   └── page.tsx          # Upload & dataset management
│   │   │   ├── dashboard/preprocess/
│   │   │   │   └── page.tsx          # Pre-processing job runner + EDA charts
│   │   │   ├── dashboard/train/
│   │   │   │   └── page.tsx          # Model training config + progress
│   │   │   ├── dashboard/evaluate/
│   │   │   │   └── page.tsx          # Metrics table + actual vs predicted
│   │   │   ├── dashboard/forecast/
│   │   │   │   └── page.tsx          # Forecast input + AreaChart
│   │   │   └── dashboard/ai/
│   │   │       └── page.tsx          # AI assistant (Low Priority sprint)
│   │   └── api/                      # Next.js API routes (proxy layer)
│   │       ├── datasets/
│   │       │   ├── route.ts          # GET /api/datasets
│   │       │   └── [id]/
│   │       │       ├── validate/route.ts
│   │       │       └── preview/route.ts
│   │       ├── preprocess/
│   │       │   ├── route.ts          # POST /api/preprocess
│   │       │   └── [jobId]/
│   │       │       └── status/route.ts
│   │       ├── models/
│   │       │   ├── route.ts          # GET all ModelRuns
│   │       │   ├── train/route.ts    # POST /api/models/train
│   │       │   ├── tune/route.ts     # POST /api/models/tune
│   │       │   └── [modelId]/
│   │       │       └── evaluate/route.ts
│   │       ├── forecast/
│   │       │   └── route.ts          # POST /api/forecast
│   │       └── ai/
│   │           └── chat/route.ts     # POST /api/ai/chat (AI SDK v6)
│   ├── components/
│   │   ├── ui/                       # shadcn/ui base components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── table.tsx
│   │   │   ├── select.tsx
│   │   │   ├── badge.tsx
│   │   │   └── skeleton.tsx
│   │   ├── charts/                   # Recharts wrappers
│   │   │   ├── ForecastChart.tsx     # AreaChart for forecast output
│   │   │   ├── ActualVsPredicted.tsx # LineChart overlay
│   │   │   ├── MetricsBar.tsx        # BarChart for feature importance
│   │   │   ├── CorrelationHeatmap.tsx
│   │   │   └── SeasonalDecomposition.tsx
│   │   ├── dashboard/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── MetricCard.tsx        # RMSE / MAE / MAPE / R² cards
│   │   │   └── ModelComparisonTable.tsx
│   │   ├── data/
│   │   │   ├── DatasetUploader.tsx   # Uploadthing drop zone
│   │   │   └── DatasetPreviewTable.tsx
│   │   ├── training/
│   │   │   ├── ModelConfigForm.tsx
│   │   │   └── TrainingProgressBar.tsx
│   │   └── ai/
│   │       └── ChatPanel.tsx         # AI SDK v6 useChat() (Low Priority)
│   ├── lib/
│   │   ├── prisma.ts                 # Prisma client singleton
│   │   ├── api-client.ts             # Typed fetch wrapper for FastAPI proxy
│   │   ├── uploadthing.ts            # Uploadthing config
│   │   └── utils.ts                  # cn(), formatKwh(), etc.
│   ├── types/
│   │   ├── dataset.ts
│   │   ├── model.ts
│   │   └── forecast.ts
│   └── hooks/
│       ├── useJobPoller.ts           # Polls /status endpoint until COMPLETE
│       └── useModels.ts
├── prisma/
│   ├── schema.prisma
│   └── migrations/
├── prisma.config.ts                  # Prisma 7 — DB connection config
├── proxy.ts                          # Clerk clerkMiddleware() (NOT middleware.ts)
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── .env.local                        # local secrets (gitignored)
├── .env.example
└── package.json
```

### 2.2 ML Backend — `ecfml-api/`

```
ecfml-api/
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point
│   ├── config.py                     # Pydantic BaseSettings
│   ├── dependencies.py               # JWT validation, shared deps
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── datasets.py               # /api/v1/datasets/*
│   │   ├── preprocessing.py          # /api/v1/preprocessing/*
│   │   ├── models.py                 # /api/v1/models/*
│   │   └── forecast.py               # /api/v1/forecast
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Pydantic request/response models
│   │   ├── preprocessing.py
│   │   ├── model.py
│   │   └── forecast.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── dataset_service.py        # File download, validation logic
│   │   ├── preprocessing_service.py  # Pipeline orchestration
│   │   ├── training_service.py       # Model training dispatcher
│   │   ├── evaluation_service.py     # Metrics computation
│   │   └── forecast_service.py       # Inference logic
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── preprocessing/
│   │   │   ├── imputer.py            # Missing value imputation
│   │   │   ├── outlier.py            # Z-score + IQR detection
│   │   │   ├── features.py           # Feature engineering
│   │   │   ├── scaler.py             # MinMaxScaler wrapper
│   │   │   └── splitter.py           # Train/val/test split
│   │   ├── models/
│   │   │   ├── ann.py                # ANN (TensorFlow/Keras)
│   │   │   ├── lstm.py               # LSTM (TensorFlow/Keras)
│   │   │   ├── random_forest.py      # RF (scikit-learn)
│   │   │   └── svr.py                # SVR (scikit-learn)
│   │   ├── evaluation/
│   │   │   └── metrics.py            # RMSE, MAE, MAPE, R²
│   │   └── persistence/
│   │       └── model_store.py        # save/load .h5 and .joblib files
│   └── utils/
│       ├── logger.py                 # Structured logging
│       └── job_store.py              # In-memory job status dict
├── data/
│   ├── raw/                          # Downloaded from Uploadthing URLs
│   └── processed/                    # Post-preprocessing datasets
├── models/                           # Saved .h5 and .joblib files
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
├── requirements.txt
├── .env
├── .env.example
├── Dockerfile
└── render.yaml
```

---

## 3. System Architecture

### 3.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          BROWSER                                 │
│   Next.js 16 App (React 19.2 + Tailwind + shadcn/ui + Recharts) │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VERCEL (Next.js 16 Server)                       │
│                                                                   │
│  ┌─────────────────┐   ┌───────────────────┐                     │
│  │  App Router RSC  │   │   /api/* Routes    │                    │
│  │  (page.tsx)      │   │   (proxy layer)    │                    │
│  └────────┬────────┘   └────────┬──────────┘                     │
│           │                     │                                 │
│  ┌────────▼────────┐   ┌────────▼──────────┐                     │
│  │   Prisma 7      │   │  Clerk JWT inject  │                     │
│  │  @prisma/adapter│   │  + fetch to FastAPI│                     │
│  │  -pg            │   └────────┬──────────┘                     │
│  └────────┬────────┘            │ HTTPS + Bearer token            │
└───────────┼─────────────────────┼───────────────────────────────┘
            │                     │
            ▼                     ▼
┌───────────────────┐   ┌─────────────────────────────────────────┐
│  NEON POSTGRES    │   │         RENDER (FastAPI Service)          │
│  (Serverless PG)  │   │                                           │
│                   │   │  ┌──────────────────────────────────────┐ │
│  Users            │   │  │  FastAPI + Uvicorn ASGI               │ │
│  Datasets         │   │  │  ┌──────────┐  ┌──────────────────┐  │ │
│  PreprocessJobs   │   │  │  │  Routers  │  │  BackgroundTasks  │ │ │
│  ModelRuns        │   │  │  └────┬─────┘  └────────┬─────────┘  │ │
│  EvalResults      │   │  │       │                  │             │ │
│  ForecastRuns     │   │  │  ┌────▼─────┐  ┌────────▼─────────┐  │ │
└───────────────────┘   │  │  │ Services  │  │    ML Pipeline   │  │ │
                        │  │  └──────────┘  │  ANN/LSTM/RF/SVR │  │ │
                        │  │                └──────────────────┘  │ │
                        │  └──────────────────────────────────────┘ │
                        │                                            │
                        │  ┌──────────────────────────────────────┐ │
                        │  │  Filesystem Storage                    │ │
                        │  │  /data/raw   /data/processed           │ │
                        │  │  /models     (.h5, .joblib, scalers)   │ │
                        │  └──────────────────────────────────────┘ │
                        └─────────────────────────────────────────┘

                   ┌──────────────┐    ┌───────────────┐
                   │    CLERK v7   │    │  UPLOADTHING   │
                   │  (Auth/JWT)   │    │ (File Storage) │
                   └──────────────┘    └───────────────┘
```

### 3.2 Authentication Flow

```
1. User hits /dashboard
2. proxy.ts → clerkMiddleware() checks session
3. No session → redirect to /sign-in
4. Clerk issues JWT on sign-in
5. auth() in RSC → userId for Prisma queries
6. For FastAPI calls:
   Next.js API route extracts JWT via auth()
   Attaches as Authorization: Bearer <token>
   FastAPI validates JWT signature against Clerk JWKS endpoint
```

### 3.3 Request Lifecycle (Training Example)

```
User clicks "Start Training"
       │
       ▼
ModelConfigForm.tsx (Client Component)
  POST /api/models/train  { modelType, datasetId, hyperparams }
       │
       ▼
/api/models/train/route.ts  (Next.js API Route)
  1. auth() → get userId + JWT
  2. Validate request body
  3. POST to FastAPI /api/v1/models/train with Bearer token
       │
       ▼
FastAPI POST /api/v1/models/train
  1. Validate JWT (Clerk JWKS)
  2. Create ModelRun record (via HTTP call back to Next.js /api/internal/model-runs OR direct Neon connection)
  3. Enqueue BackgroundTask: training_service.train(job_id, config)
  4. Return { job_id, status: "PENDING" } immediately
       │
       ▼
Frontend: useJobPoller(job_id)
  GET /api/models/jobs/{job_id}/status every 3s
       │
       ▼
FastAPI GET /api/v1/models/jobs/{job_id}/status
  Returns { status, progress, epoch, loss } from in-memory job_store
       │
       ▼
Training completes → FastAPI saves model to disk
  Updates job_store status to COMPLETE
  Next polling call returns COMPLETE + evaluation metadata
       │
       ▼
Frontend stops polling, shows success state
```

---

## 4. Component Breakdown

### 4.1 Next.js Frontend Components

#### `proxy.ts` — Authentication Gate
```typescript
// proxy.ts  (NOT middleware.ts — Next.js 16 requirement)
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'

const isPublicRoute = createRouteMatcher([
  '/',
  '/sign-in(.*)',
  '/sign-up(.*)',
])

export const proxy = clerkMiddleware(async (auth, request) => {
  if (!isPublicRoute(request)) {
    await auth.protect()
  }
})

export const config = {
  matcher: ['/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)', '/(api|trpc)(.*)'],
}
```

#### `lib/prisma.ts` — Prisma 7 Singleton
```typescript
// lib/prisma.ts
import { PrismaClient } from '@prisma/client'
import { PrismaNeon } from '@prisma/adapter-neon'
import { Pool } from '@neondatabase/serverless'

const pool = new Pool({ connectionString: process.env.DATABASE_URL })
const adapter = new PrismaNeon(pool)

const globalForPrisma = globalThis as unknown as { prisma: PrismaClient }

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({ adapter })

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma
```

#### `lib/api-client.ts` — FastAPI Proxy Helper
```typescript
// lib/api-client.ts
import { auth } from '@clerk/nextjs/server'

const FASTAPI_BASE = process.env.FASTAPI_URL ?? 'http://localhost:8000'

export async function mlFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const { getToken } = await auth()
  const token = await getToken()

  const res = await fetch(`${FASTAPI_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  })

  if (!res.ok) {
    const error = await res.json()
    throw new Error(error.detail ?? `FastAPI error ${res.status}`)
  }

  return res.json() as Promise<T>
}
```

#### `hooks/useJobPoller.ts` — Background Job Status Poller
```typescript
// hooks/useJobPoller.ts
'use client'
import { useEffect, useState } from 'react'

type JobStatus = 'PENDING' | 'RUNNING' | 'COMPLETE' | 'FAILED' | 'WARNING'

interface JobState {
  status: JobStatus
  progress?: number
  epoch?: number
  loss?: number
  error?: string
}

export function useJobPoller(jobId: string | null, intervalMs = 3000) {
  const [state, setState] = useState<JobState>({ status: 'PENDING' })

  useEffect(() => {
    if (!jobId) return
    const interval = setInterval(async () => {
      const res = await fetch(`/api/models/jobs/${jobId}/status`)
      const data: JobState = await res.json()
      setState(data)
      if (data.status === 'COMPLETE' || data.status === 'FAILED') {
        clearInterval(interval)
      }
    }, intervalMs)
    return () => clearInterval(interval)
  }, [jobId, intervalMs])

  return state
}
```

#### `components/charts/ForecastChart.tsx`
```typescript
// components/charts/ForecastChart.tsx
'use client'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'

interface ForecastPoint {
  timestamp: string
  predicted: number
  actual?: number
}

interface ForecastChartProps {
  data: ForecastPoint[]
  title?: string
}

export function ForecastChart({ data, title }: ForecastChartProps) {
  return (
    <div className="w-full">
      {title && <h3 className="text-sm font-medium mb-3 text-muted-foreground">{title}</h3>}
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="predicted" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#2E75B6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#2E75B6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" tick={{ fontSize: 11 }} />
          <YAxis tickFormatter={(v) => `${v} kWh`} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v: number) => [`${v.toFixed(2)} kWh`]} />
          <Legend />
          {data[0]?.actual !== undefined && (
            <Area type="monotone" dataKey="actual" stroke="#888" fill="none"
              strokeDasharray="4 4" name="Actual" />
          )}
          <Area type="monotone" dataKey="predicted" stroke="#2E75B6"
            fill="url(#predicted)" name="Predicted" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
```

---

## 5. Database Schema (Prisma 7)

### 5.1 `prisma.config.ts`
```typescript
// prisma.config.ts  — Prisma 7 requirement (replaces datasource in schema.prisma)
import { defineConfig } from 'prisma/config'
import { Pool } from '@neondatabase/serverless'
import 'dotenv/config'

export default defineConfig({
  earlyAccess: true,
  schema: 'prisma/schema.prisma',
  migrate: {
    async adapter() {
      const { PrismaNeon } = await import('@prisma/adapter-neon')
      const pool = new Pool({ connectionString: process.env.DATABASE_URL! })
      return new PrismaNeon(pool)
    },
  },
})
```

### 5.2 `prisma/schema.prisma`
```prisma
// prisma/schema.prisma
// Note: No datasource block in Prisma 7 — connection is in prisma.config.ts

generator client {
  provider        = "prisma-client"
  output          = "../node_modules/.prisma/client"
}

// ─── User ──────────────────────────────────────────────────────────────────
// userId mirrors the Clerk user ID (cuid format)
model User {
  id        String   @id           // Clerk userId
  email     String   @unique
  createdAt DateTime @default(now())

  datasets         Dataset[]
  weatherDatasets  WeatherDataset[]
  preprocessJobs   PreprocessingJob[]
  modelRuns        ModelRun[]
  forecastRuns     ForecastRun[]
}

// ─── Dataset ────────────────────────────────────────────────────────────────
model Dataset {
  id               String   @id @default(cuid())
  userId           String
  user             User     @relation(fields: [userId], references: [id])
  name             String
  uploadthingUrl   String
  uploadthingKey   String
  rowCount         Int?
  validationStatus ValidationStatus @default(PENDING)
  validationReport Json?            // { missing_columns, null_pct, warnings }
  createdAt        DateTime @default(now())
  deletedAt        DateTime?        // soft delete

  weatherDatasets  WeatherDataset[]
  preprocessJobs   PreprocessingJob[]
}

enum ValidationStatus {
  PENDING
  VALID
  INVALID
  WARNING
}

// ─── WeatherDataset ─────────────────────────────────────────────────────────
model WeatherDataset {
  id             String   @id @default(cuid())
  userId         String
  user           User     @relation(fields: [userId], references: [id])
  datasetId      String
  dataset        Dataset  @relation(fields: [datasetId], references: [id])
  uploadthingUrl String
  uploadthingKey String
  createdAt      DateTime @default(now())
}

// ─── PreprocessingJob ────────────────────────────────────────────────────────
model PreprocessingJob {
  id                String    @id @default(cuid())
  userId            String
  user              User      @relation(fields: [userId], references: [id])
  datasetId         String
  dataset           Dataset   @relation(fields: [datasetId], references: [id])
  status            JobStatus @default(PENDING)
  splitRatioTrain   Float     @default(0.70)
  splitRatioVal     Float     @default(0.15)
  splitRatioTest    Float     @default(0.15)
  resultSummaryJson Json?     // { row_count, null_pct, outlier_count, feature_list }
  edaChartsJson     Json?     // chart data arrays for Recharts
  processedFilePath String?   // path on FastAPI filesystem
  createdAt         DateTime  @default(now())
  updatedAt         DateTime  @updatedAt

  modelRuns ModelRun[]
}

enum JobStatus {
  PENDING
  RUNNING
  COMPLETE
  FAILED
  WARNING
}

// ─── ModelRun ────────────────────────────────────────────────────────────────
model ModelRun {
  id                  String          @id @default(cuid())
  userId              String
  user                User            @relation(fields: [userId], references: [id])
  preprocessJobId     String
  preprocessJob       PreprocessingJob @relation(fields: [preprocessJobId], references: [id])
  modelType           ModelType
  hyperparamsJson     Json            // { layers, units, lr, epochs, etc. }
  status              JobStatus       @default(PENDING)
  modelFilePath       String?         // .h5 or .joblib path on FastAPI server
  scalerFilePath      String?         // fitted MinMaxScaler path
  trainingTimeSecs    Float?
  createdAt           DateTime        @default(now())

  evaluationResult EvaluationResult?
  forecastRuns     ForecastRun[]
}

enum ModelType {
  ANN
  LSTM
  RANDOM_FOREST
  SVR
}

// ─── EvaluationResult ───────────────────────────────────────────────────────
model EvaluationResult {
  id           String   @id @default(cuid())
  modelRunId   String   @unique
  modelRun     ModelRun @relation(fields: [modelRunId], references: [id])
  rmse         Float
  mae          Float
  mape         Float
  r2           Float
  testSetSize  Int
  actualJson   Json     // [{ timestamp, value }] for Recharts
  predictedJson Json    // [{ timestamp, value }] for Recharts
  featureImportanceJson Json? // RF only: [{ feature, importance }]
  evaluatedAt  DateTime @default(now())
}

// ─── ForecastRun ─────────────────────────────────────────────────────────────
model ForecastRun {
  id           String      @id @default(cuid())
  userId       String
  user         User        @relation(fields: [userId], references: [id])
  modelRunId   String
  modelRun     ModelRun    @relation(fields: [modelRunId], references: [id])
  startDate    DateTime
  horizonDays  Int
  resolution   Resolution
  forecastJson Json        // [{ timestamp, value }]
  createdAt    DateTime    @default(now())
}

enum Resolution {
  HOURLY
  DAILY
  WEEKLY
}
```

### 5.3 Key Indexes to Add via Migration
```prisma
// Add to schema after initial migration:
@@index([userId, createdAt])  // on Dataset, ModelRun, ForecastRun
@@index([status])             // on PreprocessingJob, ModelRun
```

---

## 6. API Design (FastAPI)

### 6.1 `app/main.py`
```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import datasets, preprocessing, models, forecast
from app.config import settings

app = FastAPI(
    title="ECFML ML Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Vercel domain only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router,      prefix="/api/v1")
app.include_router(preprocessing.router, prefix="/api/v1")
app.include_router(models.router,        prefix="/api/v1")
app.include_router(forecast.router,      prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ecfml-api"}
```

### 6.2 `app/config.py`
```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CLERK_JWKS_URL: str           # https://clerk.your-domain.com/.well-known/jwks.json
    ALLOWED_ORIGINS: list[str]    # ["https://ecfml.vercel.app"]
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 6.3 `app/dependencies.py` — JWT Validation
```python
# app/dependencies.py
import httpx
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    token = credentials.credentials
    try:
        # Fetch Clerk JWKS and validate
        async with httpx.AsyncClient() as client:
            jwks = (await client.get(settings.CLERK_JWKS_URL)).json()
        payload = jwt.decode(token, jwks, algorithms=["RS256"])
        return {"user_id": payload.get("sub")}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
```

### 6.4 Datasets Router
```
GET    /api/v1/datasets/{id}/preview
       Query: ?rows=100
       Response: { columns: str[], rows: list[dict], total_rows: int }

POST   /api/v1/datasets/{id}/validate
       Body:   { file_url: str }
       Response: { valid: bool, missing_columns: list[str],
                   row_count: int, null_pct: float, warnings: list[str] }
```

### 6.5 Preprocessing Router
```
POST   /api/v1/preprocessing/run
       Body:   { job_id: str, dataset_url: str, weather_url: str | null,
                 split_train: float, split_val: float, split_test: float }
       Response: { job_id: str, status: "PENDING" }

GET    /api/v1/preprocessing/{job_id}/status
       Response: { job_id: str, status: JobStatus, progress: float,
                   summary: PreprocessSummary | null }

GET    /api/v1/preprocessing/{job_id}/eda
       Response: { distributions: ChartData, correlation: ChartData,
                   seasonal: ChartData, lag_acf: ChartData }
```

### 6.6 Models Router
```
POST   /api/v1/models/train
       Body:   { job_id: str, preprocess_job_id: str, model_type: ModelType,
                 hyperparams: ModelHyperparams }
       Response: { job_id: str, status: "PENDING" }

POST   /api/v1/models/tune
       Body:   { job_id: str, preprocess_job_id: str, model_type: ModelType,
                 param_grid: dict }
       Response: { job_id: str, status: "PENDING" }

GET    /api/v1/models/jobs/{job_id}/status
       Response: { job_id: str, status: JobStatus, progress: float,
                   epoch: int | null, loss: float | null, val_loss: float | null }

POST   /api/v1/models/{model_id}/evaluate
       Body:   { model_run_id: str }
       Response: { model_run_id: str, rmse: float, mae: float,
                   mape: float, r2: float, test_set_size: int,
                   actual: list[TimePoint], predicted: list[TimePoint],
                   feature_importance: list[FeatureScore] | null }
```

### 6.7 Forecast Router
```
POST   /api/v1/forecast
       Body:   { forecast_id: str, model_run_id: str,
                 start_date: str (ISO8601), horizon_days: int,
                 resolution: "hourly" | "daily" | "weekly",
                 weather_overrides: dict | null }
       Response: { forecast_id: str,
                   predictions: list[{ timestamp: str, value: float }],
                   model_run_id: str, horizon_days: int }
```

### 6.8 Pydantic Schemas

```python
# app/schemas/model.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class ModelType(str, Enum):
    ANN = "ANN"
    LSTM = "LSTM"
    RANDOM_FOREST = "RANDOM_FOREST"
    SVR = "SVR"

class ANNHyperparams(BaseModel):
    layers: list[int] = Field(default=[64, 32])   # neurons per layer
    activation: str = "relu"
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32

class LSTMHyperparams(BaseModel):
    time_steps: int = 168              # lookback window in hours
    units: list[int] = Field(default=[64, 32])
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32

class RFHyperparams(BaseModel):
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    n_jobs: int = -1

class SVRHyperparams(BaseModel):
    kernel: str = "rbf"
    C: float = 1.0
    epsilon: float = 0.1
    gamma: str = "scale"

class TrainRequest(BaseModel):
    job_id: str
    preprocess_job_id: str
    model_type: ModelType
    hyperparams: ANNHyperparams | LSTMHyperparams | RFHyperparams | SVRHyperparams

class EvaluationResponse(BaseModel):
    model_run_id: str
    rmse: float
    mae: float
    mape: float
    r2: float
    test_set_size: int
    actual: list[dict]       # [{timestamp, value}]
    predicted: list[dict]    # [{timestamp, value}]
    feature_importance: Optional[list[dict]] = None  # RF only
```

---

## 7. ML Pipeline Design

### 7.1 Pipeline Overview

```
Raw CSV/Excel (from Uploadthing URL)
       │
       ▼
┌─────────────────────────────────────────────────┐
│              PRE-PROCESSING PIPELINE              │
│                                                   │
│  1. Load & parse (pandas read_csv / read_excel)   │
│  2. Parse timestamps → DatetimeIndex              │
│  3. Resample to hourly frequency                  │
│  4. Merge weather data on timestamp               │
│  5. Impute missing values                         │
│     ├── gaps ≤ 6h  → linear interpolation         │
│     └── gaps > 6h  → forward fill                 │
│  6. Detect & flag outliers (Z-score + IQR)        │
│  7. Engineer features (see 7.2)                   │
│  8. Fit MinMaxScaler on training split only       │
│  9. Transform all splits with fitted scaler       │
│  10. Save: processed.parquet + scaler.joblib      │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│                MODEL TRAINING                     │
│                                                   │
│  Load processed.parquet                           │
│  Split: 70% train / 15% val / 15% test           │
│         (time-ordered, NO shuffling)              │
│                                                   │
│  For ANN / SVR / RF:                              │
│    X shape: (samples, n_features)                 │
│    y shape: (samples,)                            │
│                                                   │
│  For LSTM:                                        │
│    Build sliding window sequences                 │
│    X shape: (samples, time_steps, n_features)     │
│    y shape: (samples,)                            │
│                                                   │
│  Train → Validate → Save model file               │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│                  EVALUATION                       │
│                                                   │
│  Load model file + scaler                         │
│  Predict on test split                            │
│  Inverse-transform predictions → real kWh values │
│  Compute: RMSE, MAE, MAPE, R²                     │
│  Return: actual[], predicted[], feature_imp[]     │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│                   INFERENCE                       │
│                                                   │
│  Load model file + scaler                         │
│  Build input features for forecast horizon        │
│  Iterative prediction (rolling forecast):         │
│    predict t+1 → append to history → predict t+2  │
│  Inverse-transform → return predictions[]         │
└─────────────────────────────────────────────────┘
```

### 7.2 Engineered Features

| Feature | Description | Code |
|---|---|---|
| `hour` | Hour of day (0–23) | `df.index.hour` |
| `day_of_week` | Day (0=Mon, 6=Sun) | `df.index.dayofweek` |
| `month` | Month (1–12) | `df.index.month` |
| `quarter` | Quarter (1–4) | `df.index.quarter` |
| `is_weekend` | 1 if Sat/Sun | `(df.index.dayofweek >= 5).astype(int)` |
| `is_holiday` | 1 if public holiday | merge with holidays CSV |
| `season` | 0–3 (dry/rainy) | custom Cameroon seasons |
| `lag_1h` | Consumption t-1h | `df['consumption'].shift(1)` |
| `lag_24h` | Consumption t-24h | `df['consumption'].shift(24)` |
| `lag_48h` | Consumption t-48h | `df['consumption'].shift(48)` |
| `lag_168h` | Same hour last week | `df['consumption'].shift(168)` |
| `rolling_mean_24h` | 24h rolling mean | `.rolling(24).mean()` |
| `rolling_std_24h` | 24h rolling std | `.rolling(24).std()` |
| `rolling_mean_7d` | 7-day rolling mean | `.rolling(168).mean()` |
| `sin_hour` | Daily Fourier sin | `sin(2π × hour / 24)` |
| `cos_hour` | Daily Fourier cos | `cos(2π × hour / 24)` |
| `sin_year` | Annual Fourier sin | `sin(2π × doy / 365)` |
| `cos_year` | Annual Fourier cos | `cos(2π × doy / 365)` |
| `temperature_c` | Hourly temperature | weather merge |
| `humidity_pct` | Hourly humidity | weather merge |
| `rainfall_mm` | Hourly rainfall | weather merge |

### 7.3 Model Implementations

#### ANN (`app/ml/models/ann.py`)
```python
from tensorflow import keras
from app.schemas.model import ANNHyperparams

def build_ann(n_features: int, params: ANNHyperparams) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features,)))
    for units in params.layers:
        model.add(keras.layers.Dense(units, activation=params.activation))
        model.add(keras.layers.Dropout(params.dropout))
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

def train_ann(X_train, y_train, X_val, y_val, params: ANNHyperparams,
              job_id: str, job_store: dict):
    model = build_ann(X_train.shape[1], params)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: job_store.update({
                job_id: { "status": "RUNNING",
                          "epoch": epoch + 1,
                          "loss": round(logs["loss"], 6),
                          "val_loss": round(logs.get("val_loss", 0), 6),
                          "progress": round((epoch + 1) / params.epochs * 100, 1) }
            })
        )
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=params.epochs, batch_size=params.batch_size,
              callbacks=callbacks, verbose=0)
    return model
```

#### LSTM (`app/ml/models/lstm.py`)
```python
import numpy as np
from tensorflow import keras
from app.schemas.model import LSTMHyperparams

def build_sequences(X: np.ndarray, y: np.ndarray, time_steps: int):
    """Build sliding window sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)  # shapes: (N, T, F) and (N,)

def build_lstm(time_steps: int, n_features: int,
               params: LSTMHyperparams) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(time_steps, n_features)))
    for i, units in enumerate(params.units):
        return_sequences = (i < len(params.units) - 1)
        model.add(keras.layers.LSTM(units, return_sequences=return_sequences))
        model.add(keras.layers.Dropout(params.dropout))
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss='mse', metrics=['mae']
    )
    return model
```

#### Random Forest (`app/ml/models/random_forest.py`)
```python
from sklearn.ensemble import RandomForestRegressor
from app.schemas.model import RFHyperparams

def train_rf(X_train, y_train, params: RFHyperparams) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        n_jobs=params.n_jobs,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
```

#### SVR (`app/ml/models/svr.py`)
```python
from sklearn.svm import SVR
from app.schemas.model import SVRHyperparams

def train_svr(X_train, y_train, params: SVRHyperparams) -> SVR:
    model = SVR(
        kernel=params.kernel,
        C=params.C,
        epsilon=params.epsilon,
        gamma=params.gamma
    )
    model.fit(X_train, y_train)
    return model
```

### 7.4 Evaluation Metrics (`app/ml/evaluation/metrics.py`)
```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    r2   = float(r2_score(y_true, y_pred))
    return { "rmse": rmse, "mae": mae, "mape": mape, "r2": r2 }
```

### 7.5 Model Persistence (`app/ml/persistence/model_store.py`)
```python
import os, joblib
from tensorflow import keras

MODELS_DIR = os.getenv("MODELS_DIR", "./models")

def save_keras_model(model, model_run_id: str) -> str:
    path = os.path.join(MODELS_DIR, f"{model_run_id}.keras")
    model.save(path)
    return path

def load_keras_model(path: str):
    return keras.models.load_model(path)

def save_sklearn_model(model, model_run_id: str) -> str:
    path = os.path.join(MODELS_DIR, f"{model_run_id}.joblib")
    joblib.dump(model, path)
    return path

def load_sklearn_model(path: str):
    return joblib.load(path)

def save_scaler(scaler, model_run_id: str) -> str:
    path = os.path.join(MODELS_DIR, f"{model_run_id}_scaler.joblib")
    joblib.dump(scaler, path)
    return path
```

---

## 8. UI/UX Wireframes & User Flows

### 8.1 User Flow Overview

```
Landing (/) ──► Sign In ──► Dashboard Overview
                                   │
              ┌────────────────────┼─────────────────────┐
              │                    │                      │
              ▼                    ▼                      ▼
         /data              /preprocess               /train
    Upload datasets      Run pipeline             Configure + train
    Preview table        View EDA charts          Poll progress
         │                    │                      │
         └────────────────────┴──────────────────────┘
                                   │
              ┌────────────────────┼──────────────────────┐
              │                                           │
              ▼                                           ▼
         /evaluate                                   /forecast
    Compare metrics table                       Select model + params
    Actual vs predicted                         View AreaChart
    Feature importance (RF)                     Download CSV
```

### 8.2 Dashboard Layout (All Pages)

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER                                          [User Avatar] │
│  ECFML                                      [Clerk UserButton] │
├──────────────┬───────────────────────────────────────────────┤
│  SIDEBAR     │  MAIN CONTENT AREA                             │
│              │                                                │
│  ▸ Overview  │                                                │
│  ▸ Data      │                                                │
│  ▸ Preprocess│                                                │
│  ▸ Train     │                                                │
│  ▸ Evaluate  │                                                │
│  ▸ Forecast  │                                                │
│  ▸ AI Chat   │                                                │
│    (low pri) │                                                │
│              │                                                │
└──────────────┴───────────────────────────────────────────────┘
```

### 8.3 `/dashboard` — Overview Page

```
┌──────────────────────────────────────────────────────────────┐
│  Welcome back, Elvis                          May 2026        │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│  Datasets    │  Model Runs  │  Best MAPE   │  Last Forecast  │
│     3        │     7        │    7.2%      │  2 hours ago    │
│  [card]      │  [card]      │  [card]      │  [card]         │
├──────────────┴──────────────┴──────────────┴─────────────────┤
│  Recent Activity                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  LSTM training complete — MAPE 7.2%       2 hours ago   │ │
│  │  Pre-processing job complete              3 hours ago   │ │
│  │  Dataset "ENEO_2020_2025.csv" uploaded    Yesterday     │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 8.4 `/dashboard/data` — Data Management

```
┌──────────────────────────────────────────────────────────────┐
│  Data Management                                              │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Upload Consumption Dataset                              │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │  Drag & drop CSV or Excel file here                │ │ │
│  │  │           or click to browse                        │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Upload Weather Dataset (optional)                       │ │
│  │  [same drop zone]                                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Your Datasets                                                │
│  ┌─────────────┬──────────┬───────────┬──────────┬────────┐ │
│  │  Name       │  Rows    │  Status   │  Date    │ Action │ │
│  ├─────────────┼──────────┼───────────┼──────────┼────────┤ │
│  │ENEO_2020.csv│ 43,824   │ ✓ Valid   │ May 1    │[Use][⌫]│ │
│  │weather.csv  │ 43,824   │ ✓ Valid   │ May 1    │[Use][⌫]│ │
│  └─────────────┴──────────┴───────────┴──────────┴────────┘ │
│                                                               │
│  Preview: ENEO_2020.csv (first 100 rows)                     │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  timestamp        │  region  │  consumption_kwh       │   │
│  │  2020-01-01 00:00 │  NW      │  1243.5               │   │
│  │  2020-01-01 01:00 │  NW      │  1198.2               │   │
│  │  ...              │  ...     │  ...                  │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 8.5 `/dashboard/preprocess` — Pre-processing

```
┌──────────────────────────────────────────────────────────────┐
│  Pre-processing                                               │
│                                                               │
│  Dataset: [ENEO_2020.csv ▼]   Weather: [weather.csv ▼]       │
│  Split: Train [70%] Val [15%] Test [15%]    [Run Pipeline]   │
│                                                               │
│  ──── Status: COMPLETE ─────────────────────────────────     │
│  Rows processed: 43,824  │  Outliers flagged: 127            │
│  Missing values filled: 34  │  Features engineered: 21       │
│                                                               │
│  EDA Charts                                                   │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  │
│  │  Consumption Distribution│  │  Correlation Heatmap     │  │
│  │  [Recharts Histogram]    │  │  [Recharts Heatmap]      │  │
│  └──────────────────────────┘  └──────────────────────────┘  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  │
│  │  Seasonal Decomposition  │  │  Lag ACF Plot            │  │
│  │  [Recharts LineChart]    │  │  [Recharts BarChart]     │  │
│  └──────────────────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 8.6 `/dashboard/train` — Model Training

```
┌──────────────────────────────────────────────────────────────┐
│  Model Training                                               │
│                                                               │
│  Pre-processing Job: [job_abc123 ▼]                          │
│                                                               │
│  Select Model:  ○ ANN  ● LSTM  ○ Random Forest  ○ SVR        │
│                                                               │
│  LSTM Hyperparameters                                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Time Steps:   [168]    Units:       [64, 32]          │  │
│  │  Dropout:      [0.2]    Learning Rate: [0.001]         │  │
│  │  Epochs:       [50]     Batch Size:   [32]             │  │
│  │                         ☐ Enable Grid Search CV        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│                              [Start Training]                 │
│                                                               │
│  ──── Training Progress ────────────────────────────────     │
│  Epoch 23/50  ████████████░░░░░░░░  46%                      │
│  Train Loss: 0.002341   Val Loss: 0.002891                    │
│                                                               │
│  Previous Runs                                                │
│  ┌──────────┬──────────┬───────────┬──────────┬───────────┐  │
│  │  Model   │  Status  │  Time     │  Val Loss│  Action   │  │
│  ├──────────┼──────────┼───────────┼──────────┼───────────┤  │
│  │  LSTM    │ COMPLETE │  18m 32s  │  0.00289 │[Evaluate] │  │
│  │  RF      │ COMPLETE │  2m 11s   │  —       │[Evaluate] │  │
│  └──────────┴──────────┴───────────┴──────────┴───────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 8.7 `/dashboard/evaluate` — Evaluation

```
┌──────────────────────────────────────────────────────────────┐
│  Model Evaluation                                             │
│                                                               │
│  Model Comparison                                             │
│  ┌───────────┬──────────┬──────────┬──────────┬────────────┐ │
│  │  Model    │  RMSE    │  MAE     │  MAPE    │  R²        │ │
│  ├───────────┼──────────┼──────────┼──────────┼────────────┤ │
│  │★ LSTM     │  89.2    │  67.4    │  7.2%    │  0.961     │ │  ← highlighted
│  │  ANN      │  112.8   │  84.1    │  9.1%    │  0.943     │ │
│  │  RF       │  134.5   │  98.7    │  10.8%   │  0.921     │ │
│  │  SVR      │  156.3   │  118.2   │  12.4%   │  0.903     │ │
│  └───────────┴──────────┴──────────┴──────────┴────────────┘ │
│                                                               │
│  View Charts for: [LSTM ▼]                                    │
│                                                               │
│  Actual vs Predicted (Test Set)                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  [Recharts LineChart — overlaid actual/predicted lines]│  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Feature Importance (Random Forest only)                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  [Recharts HorizontalBarChart — top 10 features]       │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 8.8 `/dashboard/forecast` — Forecasting

```
┌──────────────────────────────────────────────────────────────┐
│  Forecast                                                     │
│                                                               │
│  Model:      [★ LSTM (MAPE 7.2%) ▼]                          │
│  Start Date: [2025-06-01]                                     │
│  Horizon:    [7] days                                         │
│  Resolution: ● Hourly  ○ Daily  ○ Weekly                     │
│                                                               │
│                              [Generate Forecast]              │
│                                                               │
│  Forecast: June 1–7, 2025 (Hourly)                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  [Recharts AreaChart — predicted consumption kWh]      │  │
│  │   x-axis: date/time   y-axis: kWh                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Peak predicted: 1,847 kWh (Jun 3, 18:00)                    │
│  Total predicted: 198,432 kWh                                 │
│                                                               │
│                              [Download CSV ↓]                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Data Flow Diagrams

### 9.1 Dataset Upload Flow

```
User selects file
      │
      ▼
Uploadthing React component
  → streams file to Uploadthing CDN
  → onUploadComplete callback fires
      │
      ▼
Next.js Server Action: createDataset()
  prisma.dataset.create({
    data: { userId, name, uploadthingUrl, uploadthingKey }
  })
      │
      ▼
POST /api/datasets/{id}/validate  (Next.js API route)
  → mlFetch('/api/v1/datasets/{id}/validate', { file_url })
      │
      ▼
FastAPI: downloads file from uploadthingUrl
  → validates columns, computes null_pct
  → returns ValidationResponse
      │
      ▼
Next.js Server Action: updateDatasetValidation()
  prisma.dataset.update({ validationStatus, validationReport })
      │
      ▼
Page re-renders with updated status badge
```

### 9.2 Pre-processing Flow

```
User clicks "Run Pipeline"
      │
      ▼
POST /api/preprocess
  → prisma.preprocessingJob.create({ status: PENDING })
  → mlFetch('/api/v1/preprocessing/run', { job_id, dataset_url, ... })
      │
      ▼
FastAPI returns { job_id, status: "PENDING" }
  Spawns BackgroundTask: preprocessing_service.run(job_id)
      │
      ▼
Background task runs:
  1. Downloads file from Uploadthing URL
  2. Runs full pipeline (impute → outliers → features → scale → split)
  3. Saves processed.parquet to /data/processed/{job_id}.parquet
  4. Saves scaler to /models/{job_id}_scaler.joblib
  5. Updates job_store[job_id] = { status: COMPLETE, summary: {...} }
      │
      ▼
Frontend useJobPoller polls GET /api/preprocess/{jobId}/status every 3s
  → when COMPLETE, fetches EDA data
  → renders Recharts charts
      │
      ▼
Next.js Server Action: updatePreprocessJob()
  prisma.preprocessingJob.update({ status: COMPLETE, resultSummaryJson, edaChartsJson })
```

### 9.3 Forecast Flow

```
User submits forecast form
      │
      ▼
POST /api/forecast (Next.js API route)
  → prisma.forecastRun.create({ status: PENDING... })
  → mlFetch('/api/v1/forecast', { model_run_id, start_date, horizon_days, resolution })
      │
      ▼
FastAPI:
  1. Loads model file from model_run.modelFilePath
  2. Loads scaler from model_run.scalerFilePath
  3. Builds feature vectors for forecast horizon
  4. Runs iterative prediction loop
  5. Inverse-transforms predictions → real kWh
  6. Returns { predictions: [{timestamp, value}] }
      │
      ▼
Next.js API route:
  → prisma.forecastRun.update({ forecastJson: predictions, status: COMPLETE })
  → returns predictions to frontend
      │
      ▼
Frontend renders ForecastChart (Recharts AreaChart)
User can download CSV
```

---

## 10. Deployment Architecture

### 10.1 Deployment Overview

```
                   Git Push to main
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   Vercel (auto-deploy)          Render (auto-deploy)
   Next.js 16                    FastAPI (Docker)
   Build: prisma generate         CMD: uvicorn app.main:app
          && next build                 --host 0.0.0.0
                                        --port 8000
          │                             │
          │                             │
          └──────────┬──────────────────┘
                     │
                     ▼
              Neon Postgres
              (shared DB)
```

### 10.2 Vercel Configuration (`next.config.ts`)
```typescript
// next.config.ts
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  experimental: {
    cacheComponents: true,        // Next.js 16 Cache Components
    reactCompiler: true,          // React Compiler stable in Next.js 16
  },
}

export default nextConfig
```

### 10.3 Render Dockerfile (`ecfml-api/Dockerfile`)
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for scientific Python packages
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data and model directories
RUN mkdir -p data/raw data/processed models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10.4 `render.yaml`
```yaml
services:
  - type: web
    name: ecfml-api
    env: python
    dockerfilePath: ./Dockerfile
    plan: starter          # upgrade to standard for production
    envVars:
      - key: CLERK_JWKS_URL
        sync: false        # set manually in Render dashboard
      - key: ALLOWED_ORIGINS
        value: https://ecfml.vercel.app
      - key: DATA_DIR
        value: /app/data
      - key: MODELS_DIR
        value: /app/models
    disk:
      name: model-storage
      mountPath: /app/models
      sizeGB: 5
```

### 10.5 `requirements.txt` (FastAPI service)
```
fastapi[standard]==0.136.1
uvicorn[standard]==0.34.0
pydantic-settings==2.9.1
python-jose[cryptography]==3.3.0
httpx==0.28.1
pandas==2.2.3
numpy==2.2.3
scikit-learn==1.6.1
tensorflow==2.19.0
joblib==1.4.2
pyarrow==19.0.0
openpyxl==3.1.5
python-dotenv==1.1.0
```

### 10.6 `package.json` key dependencies (Next.js service)
```json
{
  "dependencies": {
    "next": "^16.2.4",
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "@clerk/nextjs": "^7.0.0",
    "@prisma/client": "^7.0.0",
    "@prisma/adapter-neon": "^7.0.0",
    "@neondatabase/serverless": "^0.10.4",
    "recharts": "^2.15.3",
    "uploadthing": "^7.6.0",
    "@uploadthing/react": "^7.6.0",
    "ai": "^6.0.0",
    "tailwindcss": "^4.1.0",
    "class-variance-authority": "^0.7.1",
    "clsx": "^2.1.1",
    "tailwind-merge": "^3.3.0"
  },
  "devDependencies": {
    "prisma": "^7.0.0",
    "typescript": "^5.8.0",
    "@types/react": "^19.1.0",
    "@types/node": "^22.0.0"
  },
  "scripts": {
    "dev": "next dev --turbopack",
    "build": "prisma generate && next build",
    "start": "next start"
  }
}
```

---

## 11. Environment Variables

### 11.1 `ecfml-web/.env.example`
```bash
# Neon Postgres
DATABASE_URL="postgresql://user:password@host.neon.tech/ecfml?sslmode=require"

# Clerk v7
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY="pk_test_..."
CLERK_SECRET_KEY="sk_test_..."
NEXT_PUBLIC_CLERK_SIGN_IN_URL="/sign-in"
NEXT_PUBLIC_CLERK_SIGN_UP_URL="/sign-up"
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL="/dashboard"
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL="/dashboard"

# Uploadthing
UPLOADTHING_SECRET="sk_live_..."
NEXT_PUBLIC_UPLOADTHING_APP_ID="..."

# FastAPI ML Service
FASTAPI_URL="https://ecfml-api.onrender.com"

# Vercel AI SDK v6 (Low Priority sprint)
AI_GATEWAY_URL="https://ai-gateway.vercel.sh"
AI_GATEWAY_KEY="..."
```

### 11.2 `ecfml-api/.env.example`
```bash
# Clerk JWKS for JWT validation
CLERK_JWKS_URL="https://your-domain.clerk.accounts.dev/.well-known/jwks.json"

# CORS
ALLOWED_ORIGINS='["https://ecfml.vercel.app","http://localhost:3000"]'

# Storage paths
DATA_DIR="./data"
MODELS_DIR="./models"
```

---

## 12. Error Handling Strategy

### 12.1 FastAPI — RFC 7807 Error Responses
All FastAPI errors follow RFC 7807 Problem Details format:
```json
{
  "type": "https://ecfml.api/errors/validation-failed",
  "title": "Dataset Validation Failed",
  "status": 422,
  "detail": "Missing required columns: consumption_kwh, timestamp",
  "missing_columns": ["consumption_kwh"]
}
```

### 12.2 Next.js — API Route Error Handling
```typescript
// Pattern for all /api/* routes
export async function POST(request: Request) {
  try {
    const body = await request.json()
    const result = await mlFetch('/api/v1/...', { method: 'POST', body: JSON.stringify(body) })
    return Response.json(result)
  } catch (error) {
    console.error('[API Route Error]', error)
    return Response.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
```

### 12.3 Frontend — Error States
- All data-fetching components use React `error.tsx` boundaries per route segment
- Training/preprocessing failures surface as toast notifications with the FastAPI `detail` message
- Empty states (no datasets, no model runs) show contextual prompts with navigation links

### 12.4 ML Pipeline — Graceful Failures
| Scenario | Behaviour |
|---|---|
| Outlier rate > 5% | Job status → WARNING; frontend prompts user to confirm |
| Training OOM error | Job status → FAILED; error message includes memory suggestion |
| MAPE > 20% after training | Warning flag on EvaluationResult; dashboard shows alert |
| Missing weather data | Pre-processing continues without weather features; logs warning |
| Model file missing at inference | FastAPI returns 404 with `"Model file not found — retrain required"` |

---

*End of ECFML System Design Document — v1.0*
