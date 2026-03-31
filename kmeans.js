/**
 * kmeans.js — K-Means clustering engine
 * Données réelles : pdv_all_pdv_all.csv  (60 536 PDV, Cluster ID -1 exclus)
 * 4 clusters : Champions | Actifs Modérés Hiver | PDV Dormants | Actifs Faibles Été
 */

"use strict";

// ─────────────────────────────────────────────
// SEGMENTS RÉELS — Cluster ID 0..3
// ─────────────────────────────────────────────
const SEGMENT_MAP = {
  0: { label: "Champions",            color: "#00d4ff" },
  1: { label: "Actifs Modérés Hiver", color: "#ffb830" },
  2: { label: "PDV Dormants",         color: "#ff5f5f" },
  3: { label: "Actifs Faibles Été",   color: "#39d98a" },
};

// ─────────────────────────────────────────────
// 1. UTILITAIRES MATHÉMATIQUES
// ─────────────────────────────────────────────

function euclidean(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function mean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function std(arr) {
  if (arr.length < 2) return 1;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length) || 1;
}

// ─────────────────────────────────────────────
// 2. NORMALISATION Z-SCORE
// ─────────────────────────────────────────────

function zScoreNormalize(points) {
  if (!points.length) return { normalized: [], means: [], stds: [] };
  const D = points[0].length;
  const means = [], stds = [];

  for (let d = 0; d < D; d++) {
    const col = points.map(p => p[d]);
    means.push(mean(col));
    stds.push(std(col));
  }

  const normalized = points.map(p =>
    p.map((v, d) => (v - means[d]) / stds[d])
  );

  return { normalized, means, stds };
}

// ─────────────────────────────────────────────
// 3. INITIALISATION K-MEANS++
// ─────────────────────────────────────────────

function kMeansPlusPlusInit(points, k) {
  const n = points.length;
  const centroids = [[...points[Math.floor(Math.random() * n)]]];

  for (let c = 1; c < k; c++) {
    const distances = points.map(p =>
      Math.min(...centroids.map(cent => euclidean(p, cent))) ** 2
    );
    const total = distances.reduce((s, d) => s + d, 0);
    let rand = Math.random() * total, chosen = 0;
    for (let i = 0; i < n; i++) {
      rand -= distances[i];
      if (rand <= 0) { chosen = i; break; }
    }
    centroids.push([...points[chosen]]);
  }

  return centroids;
}

// ─────────────────────────────────────────────
// 4. K-MEANS PRINCIPAL  (k fixé à 4)
// ─────────────────────────────────────────────

function kMeans(points, k = 4, options = {}) {
  const { maxIter = 300, tolerance = 1e-4, nInit = 5 } = options;
  const n = points.length;
  const D = points[0].length;
  k = Math.min(k, n - 1);

  let bestResult = null;

  for (let init = 0; init < nInit; init++) {
    let centroids = kMeansPlusPlusInit(points, k);
    let labels    = new Array(n).fill(0);
    let iter      = 0;

    for (iter = 0; iter < maxIter; iter++) {
      const newLabels = points.map(p => {
        let minDist = Infinity, cluster = 0;
        for (let c = 0; c < k; c++) {
          const d = euclidean(p, centroids[c]);
          if (d < minDist) { minDist = d; cluster = c; }
        }
        return cluster;
      });

      const newCentroids = Array.from({ length: k }, () => new Array(D).fill(0));
      const counts = new Array(k).fill(0);

      for (let i = 0; i < n; i++) {
        const c = newLabels[i];
        counts[c]++;
        for (let d = 0; d < D; d++) newCentroids[c][d] += points[i][d];
      }

      for (let c = 0; c < k; c++) {
        if (counts[c] === 0) {
          newCentroids[c] = [...points[Math.floor(Math.random() * n)]];
        } else {
          for (let d = 0; d < D; d++) newCentroids[c][d] /= counts[c];
        }
      }

      const shift = centroids.reduce((s, c, ci) =>
        s + euclidean(c, newCentroids[ci]), 0);
      labels    = newLabels;
      centroids = newCentroids;
      if (shift < tolerance) break;
    }

    const wcss = labels.reduce((s, c, i) =>
      s + euclidean(points[i], centroids[c]) ** 2, 0);

    if (!bestResult || wcss < bestResult.inertia) {
      bestResult = { labels, centroids, inertia: wcss, iterations: iter + 1 };
    }
  }

  return bestResult;
}

// ─────────────────────────────────────────────
// 5. MÉTHODE DU COUDE
// ─────────────────────────────────────────────

function elbowMethod(points, kMin = 2, kMax = 7) {
  const results = [];
  for (let k = kMin; k <= kMax; k++) {
    const { inertia } = kMeans(points, k, { nInit: 2, maxIter: 100 });
    results.push({ k, inertia });
  }
  return results;
}

// ─────────────────────────────────────────────
// 6. SILHOUETTE SCORE
// ─────────────────────────────────────────────

function silhouetteScore(points, labels, sampleSize = 600) {
  const n = points.length;
  const allIdx = Array.from({ length: n }, (_, i) => i);
  const sample = n > sampleSize
    ? allIdx.sort(() => Math.random() - 0.5).slice(0, sampleSize)
    : allIdx;

  let total = 0;
  for (const i of sample) {
    const ci = labels[i];
    const same  = sample.filter(j => j !== i && labels[j] === ci);
    const other = [...new Set(sample.map(j => labels[j]))].filter(c => c !== ci);
    if (!same.length) continue;

    const a = mean(same.map(j => euclidean(points[i], points[j])));
    const b = Math.min(...other.map(c => {
      const pts = sample.filter(j => labels[j] === c);
      return pts.length ? mean(pts.map(j => euclidean(points[i], points[j]))) : Infinity;
    }));

    const s = (b - a) / Math.max(a, b);
    total += isFinite(s) ? s : 0;
  }
  return total / sample.length;
}

// ─────────────────────────────────────────────
// 7. STATS PAR CLUSTER
// ─────────────────────────────────────────────

function clusterStats(rows, labels, featureNames) {
  const k = Math.max(...labels) + 1;
  return Array.from({ length: k }, (_, c) => {
    const clusterRows = rows.filter((_, i) => labels[i] === c);
    const stat = { cluster: c, size: clusterRows.length, features: {} };
    for (const feat of featureNames) {
      const vals = clusterRows.map(r => +r[feat]).filter(v => !isNaN(v));
      if (vals.length) {
        stat.features[feat] = {
          mean: mean(vals), std: std(vals),
          min: Math.min(...vals), max: Math.max(...vals),
        };
      }
    }
    return stat;
  });
}

// ─────────────────────────────────────────────
// 8. EXPORTS
// ─────────────────────────────────────────────

if (typeof module !== "undefined" && module.exports) {
  module.exports = { SEGMENT_MAP, kMeans, kMeansPlusPlusInit, elbowMethod, silhouetteScore, clusterStats, zScoreNormalize, euclidean, mean, std };
} else if (typeof window !== "undefined") {
  window.KMeansEngine = { SEGMENT_MAP, kMeans, kMeansPlusPlusInit, elbowMethod, silhouetteScore, clusterStats, zScoreNormalize, euclidean, mean, std };
}
