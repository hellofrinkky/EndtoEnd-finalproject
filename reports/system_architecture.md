# System Architecture — Project Silent Salesman

## Architecture Diagram

```mermaid
flowchart TD
    subgraph DATA["📦 Data Layer"]
        A[data/raw/CAT_FINAL.csv] --> B[data_prep/1_clean.py]
        B --> C[data/processed/clean_cat.csv]
        C --> D[data_prep/2_target.py]
        D --> E[clean_cat.csv + target_buy]
    end

    subgraph ML["🤖 ML Pipeline"]
        E --> F[analysis/3_eda_unsupervised.py]
        E --> G[models/4_supervised.py]

        subgraph UNSUP["Unsupervised"]
            F --> F1[K-Means Clustering]
            F --> F2[PCA]
            F --> F3[Isolation Forest]
        end

        subgraph SUP["Supervised"]
            G --> G1[Logistic Regression]
            G --> G2[Random Forest ✓ Best]
            G --> G3[Gradient Boosting]
        end
    end

    subgraph OUTPUTS["📊 Outputs"]
        F1 --> O1[output_plots/cluster_profile.csv]
        F1 --> O2[output_plots/cluster_radar.png]
        F2 --> O3[output_plots/pca_scatter.png]
        F3 --> O4[output_plots/anomaly_detection.png]
        G2 --> O5[output_plots/supervised/model_comparison.csv]
        G2 --> O6[output_plots/supervised/feature_importance.csv]
    end

    subgraph DB["🗄️ Database Layer"]
        O1 --> DB1[(SQLite: analytics.db)]
        O5 --> DB1
        O6 --> DB1
    end

    subgraph BACKEND["⚙️ Backend Layer"]
        DB1 --> API[Flask API — backend/app.py]
        API --> EP1[GET /api/cluster-profile]
        API --> EP2[GET /api/model-comparison]
        API --> EP3[GET /api/descriptive-stats]
        API --> EP4[GET /api/feature-importance]
    end

    subgraph FRONTEND["🖥️ Frontend Layer"]
        API --> DASH[Dash App — dashboard/app.py]
        DASH --> P1[Page 1: Home]
        DASH --> P2[Page 2: Unsupervised Learning]
        DASH --> P3[Page 3: Supervised Learning]
        DASH --> P4[Page 4: Business Insight]
    end
```

---

## Component Breakdown

### Data Layer
| File | Role |
|---|---|
| `data/raw/CAT_FINAL.csv` | Raw survey export — read-only |
| `data_prep/1_clean.py` | Cleaning, encoding, missing value handling |
| `data_prep/2_target.py` | Creates binary `target_buy` variable |
| `data/processed/clean_cat.csv` | Clean, model-ready dataset |

### ML Layer
| File | Techniques | Outputs |
|---|---|---|
| `analysis/3_eda_unsupervised.py` | K-Means, PCA, Isolation Forest | cluster_profile.csv, plots |
| `models/4_supervised.py` | LR, RF, GB + cross-validation | model_comparison.csv, feature_importance.csv |

### Database Layer
| File | Role |
|---|---|
| `backend/db.py` | SQLite init + CSV import |
| `backend/analytics.db` | Tables: cluster_results, model_results, feature_importance, anomaly_summary |

### Backend Layer
| Endpoint | Returns |
|---|---|
| `GET /api/cluster-profile` | Cluster means per feature |
| `GET /api/model-comparison` | Model metrics (F1, Accuracy, AUC) |
| `GET /api/descriptive-stats` | Feature statistics |
| `GET /api/feature-importance` | Feature importance ranking |

### Frontend Layer
| Page | Path | Content |
|---|---|---|
| Home | `/` | Project overview, team, quick stats |
| Unsupervised | `/unsupervised` | Clustering, PCA, Anomaly KPIs + charts |
| Supervised | `/supervised` | Model comparison, Feature Importance KPIs + charts |
| Business Insight | `/business-insight` | Segment strategy, design brief, recommendations |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Dashboard | Plotly Dash + dash-bootstrap-components |
| API | Flask |
| Database | SQLite (via Python sqlite3) |
| ML | scikit-learn, pandas, numpy |
| Visualization | Plotly, matplotlib (pre-generated) |
