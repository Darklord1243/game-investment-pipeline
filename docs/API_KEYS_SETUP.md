# API Keys & Credentials Setup

How to obtain every credential this project needs, and how to store each one
safely — both for local development (`.env`) and for the weekly GitHub
Actions pipeline (repository secrets).

The variable names below are exact. Use them verbatim: they must match
`.env.example` locally and the `secrets.*` references in
`.github/workflows/weekly_pipeline.yml`.

## Contents

- [Twitch](#twitch)
- [YouTube](#youtube)
- [Reddit](#reddit)
- [AWS (S3 artifact publishing)](#aws-s3-artifact-publishing)
- [Flask secret key (local only)](#flask-secret-key-local-only)
- [Where to store each credential](#where-to-store-each-credential)
- [Security hygiene](#security-hygiene)
- [Verifying everything is wired up](#verifying-everything-is-wired-up)

---

## Twitch

**Variables:** `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`

1. Go to [dev.twitch.tv/console/apps](https://dev.twitch.tv/console/apps) and
   log in with (or create) a Twitch account.
2. **Register Your Application**:
   - Name: anything (e.g. `game-engagement-miner`)
   - OAuth Redirect URL: `http://localhost` — unused. The miner authenticates
     as itself via the OAuth **client-credentials** grant, not as a logged-in
     user, so no redirect ever fires.
   - Category: "Application Integration"
3. Copy the **Client ID** shown immediately after creation.
4. Click **New Secret** to generate the **Client Secret**. It is shown once —
   copy it now, you cannot view it again (only regenerate).
5. No scopes to configure. This app never acts on behalf of a Twitch user.

## YouTube

**Variable:** `YOUTUBE_API_KEY`

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and
   create a project (or reuse an existing one).
2. **APIs & Services → Library** → search "YouTube Data API v3" → **Enable**.
3. **APIs & Services → Credentials → Create Credentials → API key**.
4. Immediately click **Restrict Key** on the new key:
   - Under "API restrictions", select **Restrict key** and choose
     **YouTube Data API v3** only.
   - Leave application restrictions (IP/referrer) unset — GitHub Actions
     runner IPs are not static, so an IP allowlist would break CI. Restricting
     by API is the meaningful protection here.
5. Free tier is 10,000 quota units/day. A weekly run at current mining volume
   fits comfortably; re-check quota if you significantly widen scope.

## Reddit

**Variables:** `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`,
`REDDIT_PASSWORD`, `REDDIT_USER_AGENT` (optional — see below)

1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) while
   logged in to the account that will own the app.
2. Click **create another app...** at the bottom of the page.
3. Select type **script** — not "web app" or "installed app". The miner uses
   PRAW's password-grant flow, which only works with `script`-type apps.
4. Redirect URI can be `http://localhost:8080` — unused for script apps.
5. After creation:
   - The string under the app name (looks like `a1B2c3D4e5F6gH`) is
     `REDDIT_CLIENT_ID`.
   - The **secret** field is `REDDIT_CLIENT_SECRET`.
6. `REDDIT_USERNAME` / `REDDIT_PASSWORD` are the login credentials of the
   account that owns the app.

> **Two-factor authentication will break this.** Reddit's password-grant
> flow rejects any account with 2FA enabled — there is no code prompt in a
> headless CI job, so the login fails outright. Either use an account with
> 2FA off, or — recommended — create a **dedicated bot account** with 2FA off
> just for this pipeline. Do not reuse your personal Reddit password here.

`REDDIT_USER_AGENT` is optional: the miner defaults to
`GameMiningScript/0.1` if unset. Reddit's API rules ask for a descriptive,
unique user agent identifying the account, e.g.:

```
GameMiningScript/0.1 by u/your_bot_account
```

## AWS (S3 artifact publishing)

**Variables:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
`AWS_S3_BUCKET`

1. In the AWS Console, create (or identify) the **S3 bucket** the pipeline
   should publish `game_metrics.db` to. Note its name and region.
2. **IAM → Users → Create user.** Do not use your AWS root account or a
   personal admin-level IAM user for this — create a dedicated user scoped
   to this one job.
3. Attach an **inline policy** scoped to just this bucket and prefix — not
   `AmazonS3FullAccess`:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:PutObject"],
         "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/game-metrics/*"
       }
     ]
   }
   ```

4. On the user's **Security credentials** tab → **Create access key** →
   choose "Third-party service" as the use case. Copy the Access Key ID and
   Secret Access Key immediately — the secret is shown once.
5. `AWS_REGION` is the bucket's region (e.g. `us-east-1`). `AWS_S3_BUCKET` is
   just the bucket name — no `s3://` prefix, no path.
6. This is a **long-lived credential**. Rotate it periodically from the IAM
   console (create a new key, update the GitHub secret, then deactivate and
   delete the old key — no downtime). If you ever want to remove long-lived
   AWS keys from GitHub entirely, GitHub Actions supports OIDC federation
   with AWS IAM roles instead of static keys; that's a larger follow-up, not
   required to get the pipeline running today.

## Flask secret key (local only)

**Variable:** `FLASK_SECRET_KEY` — used by `src/api/webapp.py` for session
signing. Not read by the CI pipeline, only needed for running the demo
webapp locally.

Generate one and paste it into `.env`:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Where to store each credential

### Locally — `.env`

1. `cp .env.example .env`
2. Fill in each value in `.env`.
3. `.gitignore` already excludes `.env` — never `git add -f` it, and double
   check `git status` before committing if you've been editing near it.

### GitHub Actions — repository secrets

This is what actually unblocks the weekly pipeline. Go to the repo on
GitHub:

**Settings → Secrets and variables → Actions → New repository secret**

Add one secret per name below, pasting only the raw value (no quotes, no
`KEY=value` syntax — the name goes in the "Name" field, the value alone goes
in "Secret"):

```
TWITCH_CLIENT_ID
TWITCH_CLIENT_SECRET
YOUTUBE_API_KEY
REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET
REDDIT_USERNAME
REDDIT_PASSWORD
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_S3_BUCKET
```

`GITHUB_TOKEN` is provided automatically by Actions — do not add it
manually. `REDDIT_USER_AGENT` and `FLASK_SECRET_KEY` are not read by the
workflow, so they don't need to be added as repository secrets.

## Security hygiene

- **Never paste real key values into chat, an issue, a PR description, or a
  commit message** — including "just to check it looks right." If a key
  value is ever exposed that way, treat it as compromised: rotate/revoke it,
  don't try to delete the message and move on.
- **Scope every credential to the minimum it needs.** The Twitch/YouTube
  keys above are inherently scoped by the provider. For AWS specifically,
  always attach a bucket-and-prefix-scoped policy, never
  `AmazonS3FullAccess` or an admin user.
- **Use a dedicated bot account for Reddit**, not a personal account —
  easier to rotate or revoke without affecting anything else you use that
  account for.
- **Rotate on a schedule**, especially the AWS access key, since it's the
  only long-lived credential here with broad blast radius if leaked.
- **Check `git status` after any broad `git add`** before committing,
  especially around `.env` — a filename can look innocuous while its
  contents aren't.
- GitHub's secret scanning (on by default for public repos, available for
  private repos too) will flag most recognizable API key formats — including
  AWS keys — if one ever lands in a commit. Don't rely on it as your only
  safeguard, but it's a real backstop.

## Verifying everything is wired up

1. In GitHub: **Settings → Secrets and variables → Actions** — confirm all
   11 secrets listed above are present (values aren't visible again once
   saved, only names).
2. Re-enable the workflow if it shows as disabled (**Actions** tab → select
   *Weekly ETL & ML Pipeline* → **Enable workflow** if prompted).
3. Trigger a manual run: **Actions → Weekly ETL & ML Pipeline → Run
   workflow**.
4. Watch the run for:
   - `static-analysis` passes (mypy clean).
   - Steam step logs `Capping seed to 50 pending AppIDs (--limit).`
   - Twitch / YouTube / Reddit steps complete without `RuntimeError` (that
     error means a secret is missing or misnamed — check spelling against
     the list above).
   - AWS steps either publish successfully or skip cleanly if AWS secrets
     aren't fully configured yet.
   - The GitHub Release step is skipped — expected, training is still gated
     until an external review-velocity label exists.
