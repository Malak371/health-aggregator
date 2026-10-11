# Deployment

The hosted demo uses three free services. Limits below were checked in October 2026 and can change; the links go to each provider's current terms.

| Part | Service | Free-tier behaviour that matters |
| --- | --- | --- |
| API | [Render](https://render.com/docs/free) web service (Docker) | Sleeps after 15 minutes without traffic; about a minute to wake; 750 instance hours a month |
| Database | [Neon](https://neon.com/pricing) PostgreSQL | Permanent free plan, 1 GB storage per project; compute suspends after 5 minutes idle and resumes on the next query |
| Dashboard | [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) | Sleeps after 12 hours without visitors; a visitor wakes it with one click |

Alternatives considered: Render's own free PostgreSQL is deleted 30 days after creation, and Supabase's free projects pause after a week of inactivity, so neither suits a demo that must stay up unattended.

## 1. Database (Neon)

1. Create a project at <https://console.neon.tech>.
2. Copy the connection string (`postgresql://...?sslmode=require`). This is a secret: do not commit it.

## 2. API (Render)

1. At <https://dashboard.render.com>, choose **New > Blueprint** and select this repository. Render reads `render.yaml`.
2. When prompted, set:
   - `DATABASE_URL`: the Neon connection string
   - `CORS_ORIGINS`: the dashboard URL from step 3 (it can be updated later)
3. Deploy. Migrations run on start and the demo dataset is seeded. Check `https://<service>.onrender.com/health`.

## 3. Dashboard (Streamlit Community Cloud)

1. At <https://share.streamlit.io>, choose **Create app** and select this repository, branch `main`, main file `frontend/app.py`.
2. Under **Advanced settings > Secrets**, add:

   ```toml
   BACKEND_URL = "https://<service>.onrender.com"
   ```

3. Deploy, then open the app in a private browser window to confirm the demo data loads.

## Environment variables

See `.env.example` for the full list with defaults.
