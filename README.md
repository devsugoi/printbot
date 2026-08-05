# Gmail Print Bot

A bot that watches your Gmail inbox, uses Gemini (free tier) to detect
emails asking you to print an attachment, and prints them to a **Brother
DCP-J100** — but only after you confirm, either on **Discord or by
replying to the email itself**. Designed to run unattended on a
**Raspberry Pi Zero 2 W**.

## How it works

```
Gmail (poll every N seconds)
   │
   ▼
Gemini classifies: "is this a print request?" + paper size guess
   │  (tries multiple API keys / models, falling back on quota errors)
   ▼
Download attachment(s)
   │  images only → combined into one short-bond-paper PDF, one image per page
   ▼
Ask on BOTH channels: Discord embed [Print] [Cancel]  +  reply in the email thread
   │  (a generated preview PDF, if any, is attached so you can check it)
   ▼
Whichever channel responds first wins — locked so the other is a no-op
   │
   ▼
Notify BOTH channels: "approved via X — printing N copies"
   │
   ▼
Check printer is online → lp -d <printer> ... -n <copies> file
   │
   ├─ success → notify BOTH: printed ✅  ("print again" available anytime)
   └─ failure → notify BOTH: failed ❌ with the error, offer reprint
```

If a single email mixes images (always short bond paper) with a real
document that needs long bond paper, the job is split into paper-size
groups and printed one group at a time, pausing to ask you to swap the
tray between groups.

Every job is saved to `state.json` (files, paper sizes, status, copies,
email thread info), so:
- The same email is never asked about twice.
- A job can be reprinted or "printed again" at any time, even after a
  restart — via the Discord button, `!reprint <message_id>`, or just
  replying to the email thread.

## Project layout

```
printbot/
├── main.py                 # entry point
├── config.example.yaml     # copy to config.yaml and fill in
├── requirements.txt
├── src/
│   ├── config.py            # loads config.yaml (+ "ENV:VAR" secrets)
│   ├── state.py              # JSON persistence: jobs, files, paper-size groups
│   ├── gmail_client.py       # Gmail API: search, download, reply-in-thread, read replies
│   ├── ai_classifier.py      # Gemini: is-it-a-print-request + reply-intent parsing
│   ├── pdf_utils.py          # image→PDF, paper size detection for real documents
│   ├── printer.py            # CUPS `lp` wrapper (paper size, copies)
│   ├── confirmation.py       # shared approve/cancel/print logic + the "first wins" lock
│   └── discord_bot.py        # Gmail + email-reply polling, Discord UI
├── credentials/              # your OAuth + token files (gitignored)
└── jobs/                     # downloaded attachments + generated PDFs
```

## 1. Set up the Raspberry Pi

```bash
sudo apt update
sudo apt install -y cups cups-bsd python3-pip python3-venv git
sudo apt install -y libreoffice   # converts .docx/.xlsx/... attachments to PDF for printing
# Word-compatible fonts — missing fonts are the #1 cause of DOCX pagination drift
sudo apt install -y fonts-crosextra-carlito fonts-crosextra-caladea \
  fonts-liberation ttf-mscorefonts-installer
sudo fc-cache -f -v
sudo usermod -aG lpadmin $USER   # then log out/in
```

LibreOffice is required for printing office documents (Word, Excel,
PowerPoint, OpenDocument, RTF): CUPS can't print those directly, so the
bot converts them to PDF with `soffice --headless` first. On a
storage-constrained Pi, `libreoffice-writer libreoffice-calc
libreoffice-impress` covers the same formats with a smaller footprint.

Office documents are converted to PDF by LibreOffice before printing.
If pagination doesn't match Word, it's usually a **missing font** — see
[DOCX conversion inaccurate / extra pages](#docx-conversion-inaccurate--extra-pages)
in Troubleshooting. The bot uses a dedicated LibreOffice profile
(`.libreoffice-printbot/`) with inch-based layout and font embedding for
closer Word-like output.

Install the Brother DCP-J100 driver and add the printer. **The DCP-J100
is an inkjet — do NOT use `printer-driver-brlaser`**, which only supports
Brother *laser* printers. With brlaser (or any wrong driver), CUPS will
happily mark jobs "completed" while the printer sits idle and nothing
comes out. Use Brother's official DCP-J100 LPR + cupswrapper driver
(from Brother's Linux download page, or their `linux-brprinter-installer`
script).

**Raspberry Pi (ARM) caveat:** Brother ships the filter as an **i386**
binary (`/opt/brother/Printers/dcpj100/lpd/brdcpj100filter`). On
`aarch64` it fails with `Exec format error` (exit 126) unless you run it
under qemu. Do **not** `apt install libc6:i386` on Raspberry Pi OS /
Debian with the `+rpt1` libc — it conflicts with the Pi-patched
`libc6:arm64`. Use the working setup below instead.

#### Make the Brother i386 filter work on a Pi

```bash
# 1. Emulator (arm64 package — no libc conflict)
sudo apt-get install -y qemu-user qemu-user-binfmt

# 2. Extract i386 libs into a private root (avoids multiarch conflict)
sudo mkdir -p /var/tmp/brfix /opt/i386root
cd /var/tmp/brfix
sudo apt-get download gcc-13-base:i386 libc6:i386 libstdc++6:i386
for d in gcc-13-base_*.deb libc6_*.deb libstdc++6_*.deb; do
  sudo dpkg-deb -x "$d" /opt/i386root
done

# Debian usrmerge: qemu looks for /lib/ld-linux.so.2 under -L prefix
sudo mkdir -p /opt/i386root/lib
sudo ln -sfn ../usr/lib/i386-linux-gnu/ld-linux.so.2 \
  /opt/i386root/lib/ld-linux.so.2

# 3. Wrap the Brother filter so CUPS always invokes qemu
FILTER=/opt/brother/Printers/dcpj100/lpd/brdcpj100filter
# Only rename once — skip if .real already exists
if [ ! -f "$FILTER.real" ]; then
  sudo mv "$FILTER" "$FILTER.real"
fi
sudo tee "$FILTER" >/dev/null <<'EOF'
#!/bin/sh
exec /usr/bin/qemu-i386 -L /opt/i386root \
  /opt/brother/Printers/dcpj100/lpd/brdcpj100filter.real "$@"
EOF
sudo chmod +x "$FILTER"

# 4. Sanity check (bare invoke → "invalid option" / exit 2 is OK;
#    exit 126 = still can't exec; ld-linux errors = -L root incomplete)
"$FILTER"; echo exit:$?

# 5. CUPS test page — must physically print
lsusb | grep -i brother
sudo /usr/sbin/cupsenable DCPJ100
lp -d DCPJ100 /usr/share/cups/data/testprint
```

Keep `/opt/i386root` and the filter wrapper permanently. You can delete
`/var/tmp/brfix` after setup. After a reboot, if printing breaks, confirm
the wrapper and linker symlink still exist.

Alternative if you have an x86 Linux box: install the Brother driver
there, share the printer, and point the Pi's CUPS queue at that share.

Add the printer via the CUPS web UI (`http://<pi-ip>:631`) or:

```bash
# Prefer the Brother DCP-J100 PPD from the official cupswrapper package,
# not "-m everywhere", once the LPR/cupswrapper debs are installed.
lpstat -p -d
lpoptions -d DCPJ100
lpoptions -p DCPJ100 -l   # paper sizes / media names your driver supports
```

**Note:** the printer is only ever fed one of two paper choices: **short
bond paper** (8.5" x 11", i.e. Letter) or **long bond paper** (8.5" x 14",
i.e. Legal). These map to the CUPS "media" values `Letter` / `Legal` in
`src/printer.py` (`CUPS_MEDIA_NAMES`). Check `lpoptions -p <printer> -l`
and adjust that dict if your driver uses different names (e.g.
`na_letter_8.5x11in` / `na_legal_8.5x14in`).

## 2. Clone the project & install Python dependencies

```bash
git clone <your-repo-url> printbot
cd printbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Gmail API setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/),
   create a project, and enable the **Gmail API**.
2. Create an **OAuth 2.0 Client ID** of type **Desktop app**. Download the
   JSON file and save it as `credentials/credentials.json`.
3. First-time authentication opens a browser, which a headless Pi doesn't
   have. Easiest option: run the bot **once on your laptop** (same
   `credentials/credentials.json`) so the browser flow completes and a
   `token.json` is generated, then copy `credentials/token.json` to the Pi.
   Alternatively, SSH into the Pi with port forwarding
   (`ssh -L 8080:localhost:8080 pi@<ip>`) and run it there directly.
4. The token auto-refreshes after that — no browser needed again.

**The bot requests two scopes: `gmail.readonly` and `gmail.send`** — the
second one is new, needed so the bot can reply in-thread to ask for
confirmation and post results. **If you set this bot up before email
confirmation existed, delete `credentials/token.json` and redo step 3** so
a token with the new scope gets issued; the old readonly-only token will
fail when the bot tries to send.

## 4. Gemini API keys (free tier)

1. Get one or more free API keys from
   [Google AI Studio](https://aistudio.google.com/apikey). Using keys from
   a few different Google accounts/projects gives you more combined free
   quota, which is why the bot supports multiple keys.
2. List them under `gemini.api_keys` in `config.yaml` (or reference
   environment variables with `ENV:VAR_NAME`, see below).
3. List the models to try, in order, under `gemini.models`. Check
   [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)
   for the current free-tier model names and rate limits, since these
   change over time — the bot will automatically skip any model that
   returns "not found" or "overloaded" and try the next one.

**Fallback behavior:** for each Gemini call (classifying a new email, or
interpreting a confirmation reply), the bot tries every model in
`gemini.models` for the current key. If a model is missing/overloaded
(HTTP 400/404/503) it tries the next model. If the key itself is out of
quota or invalid (HTTP 403/429) it moves to the next key and starts again
from the first model. If everything fails: a new email is skipped (with a
Discord warning), and a reply that can't be classified is simply left
unread until a later poll can classify it (or you approve it the other way).

## 5. Discord bot setup

1. Create an application + bot at the
   [Discord Developer Portal](https://discord.com/developers/applications).
2. Under **Bot**, enable **Message Content Intent**.
3. Invite the bot to your server with `bot` scope and `Send Messages` /
   `Read Message History` permissions.
4. Get your own Discord **user ID** (enable Developer Mode in Discord
   settings, right-click your name → Copy User ID) and the **channel ID**
   you want it to post in (right-click channel → Copy Channel ID).
5. Only your user ID can press the Print / Cancel / Print again / Reprint
   buttons or run `!reprint` / `!status` — other members in the channel
   can see the messages but not act on them.

## 6. Email-based confirmation & who's allowed to approve

Alongside the Discord message, the bot replies **in the same email
thread** asking you to confirm, e.g.:

> I think this email is asking to print the attached file(s): scan.jpg.
> Reply with something like "yes, 2 copies" to confirm, or "no" to cancel.
> You can also confirm on Discord.

A later reply like *"yeah go ahead, 3 copies"* or *"nope, cancel that"* is
interpreted by Gemini and acted on. **Whichever channel responds first
wins** — if you tap Print in Discord, the bot won't also print because a
reply happened to arrive moments later (and vice versa); both channels
just get told the job is already being handled.

**Who counts as a valid approval by email** is controlled by two settings
in `config.yaml`:

```yaml
gmail:
  approved_reply_senders: []              # e.g. ["spouse@example.com"]
  allow_non_owner_email_approval: false
```

- **Your own address always counts** — this can't be disabled.
- If `approved_reply_senders` is **non-empty**, only those addresses (plus
  you) can approve/cancel/reprint — everyone else's replies are ignored.
  This is the safest option if specific other people should be able to
  approve prints.
- Otherwise, `allow_non_owner_email_approval: true` lets **anyone** who
  replies in the thread approve it — including whoever originally sent you
  the attachment. Leave this `false` (the default) unless you specifically
  want that; otherwise someone emailing you a file could approve printing
  it themselves without you ever seeing a request.

## 7. Configure

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`. For secrets, either paste them directly or, better,
keep them out of the file entirely using `ENV:VAR_NAME`:

```yaml
gemini:
  api_keys:
    - "ENV:GEMINI_API_KEY_1"
    - "ENV:GEMINI_API_KEY_2"
discord:
  bot_token: "ENV:DISCORD_BOT_TOKEN"
```

and set those variables (e.g. in a systemd `EnvironmentFile`, see below).

## 8. Run it

```bash
python3 main.py
```

You should see a "🖨️ Print bot is online" message in your Discord channel.

## 9. Run it automatically on boot (systemd)

`/etc/systemd/system/printbot.service`:

```ini
[Unit]
Description=Gmail Print Bot
After=network-online.target cups.service
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/printbot
EnvironmentFile=/home/pi/printbot/printbot.env
ExecStart=/home/pi/printbot/venv/bin/python3 main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

`printbot.env` (keep this file `chmod 600`):

```
GEMINI_API_KEY_1=...
GEMINI_API_KEY_2=...
DISCORD_BOT_TOKEN=...
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now printbot
sudo journalctl -u printbot -f     # logs
```

## Usage

- The bot polls Gmail every `gmail.poll_interval_seconds` seconds (default
  60) for **new** matching emails, and separately polls every pending
  job's email thread on the same interval for **replies**.
- For every new matching email, Gemini decides if it's a print request and
  guesses a paper size.
- If yes: attachments are downloaded.
  - **Images** (photos, screenshots) are always combined into one PDF,
    one image per page, scaled to fill the page, on **short bond paper**
    — this combined PDF is attached to both the Discord message and the
    email reply so you can check the bot got it right before approving.
  - **Real documents/PDFs** get their paper size from the email text if
    mentioned, otherwise auto-detected from the file's own dimensions,
    otherwise the configured default.
  - If a job ends up needing **both** paper sizes (e.g. an email with
    photos plus a legal-size form), it prints the short-bond-paper part
    first, then pauses and asks again — on both channels — for you to
    swap in long bond paper before continuing.
  - If a job needs **long bond paper**, you can instead choose **Print on
    short bond** (Discord button) or reply with something like "yes, use
    short bond" — long-sized PDFs are scaled to fit letter paper so you
    don't have to swap the tray.
- You get a Discord message with **Print** / **Cancel** buttons (plus
  **Print on short bond** when long paper is needed), and a reply in the
  email thread with the same information. Clicking **Print** (or replying
  "yes") opens/asks for **how many copies** — leave it blank to default to
  1.
- Whichever channel you respond on first is honored; both channels then
  get an "approved via X — printing N copies" notice.
- If printing fails (including if the printer isn't detected at all), you
  get the specific error on **both** channels plus a **Reprint** button;
  replying to the email also retries.
- After a successful print, both channels offer **Print again** (asks for
  copies again) any time — not just right after printing.
- `!reprint <message_id> [copies]` — manually trigger a (re)print of any
  past job (IDs shown in `!status`).
- `!status` — list your 10 most recent jobs, their status, and copy count.
- **There's no time limit on confirming.** A job just sits at "awaiting
  confirmation" until you respond — by email or Discord — whenever you get
  to it. Restarting the bot doesn't lose anything either: on startup it
  re-registers Discord buttons for every pending/printed/failed job within
  `processed_email_retention_days` (so old "Print"/"Cancel"/"Print on short
  bond"/"Reprint" clicks keep working), and if it happens to restart
  mid-print, that job is marked failed on the next startup so it's
  reprintable rather than stuck.

## Customization notes

- **Search query**: tighten `gmail.search_query` (e.g. add
  `from:family@example.com`) to reduce how much gets sent to Gemini and
  avoid burning through free-tier quota on irrelevant mail.
- **Paper sizes**: the bot is intentionally locked to two choices (short
  bond paper / long bond paper) to match a printer loaded with only those
  two trays/stacks. Generated (image-combined) PDFs are always short bond
  paper by design. If you ever need more paper options, add entries to
  `printer.supported_paper_sizes`, `pdf_utils.PAPER_SIZES` /
  `PAPER_SIZE_LABELS`, and `printer.CUPS_MEDIA_NAMES` together, and update
  the allowed values in `ai_classifier.CLASSIFY_PROMPT_TEMPLATE`.
- **Multiple attachments to print selectively**: currently the bot prints
  *all* attachments on a matched email. If you want Gemini to pick specific
  files, extend the JSON schema in `ai_classifier.CLASSIFY_PROMPT_TEMPLATE`
  with a `target_attachments` field and filter in
  `discord_bot.PrintBot._prepare_job`.
- **Office attachments** (`.doc`, `.docx`, `.odt`, `.rtf`, `.xls`,
  `.xlsx`, `.ods`, `.ppt`, `.pptx`, `.odp`) are converted to PDF with
  headless LibreOffice (dedicated profile + `writer_pdf_Export` with font
  embedding) before printing, and the converted PDF is attached to the
  confirmation message so you can check the rendering before approving.
  This requires LibreOffice and Word-compatible fonts on the Pi (see setup
  step 1); if either is missing or conversion fails, the job fails with a
  clear message and can be reprinted after fixing the issue. Conversion
  also runs at print time for jobs prepared before this feature (or whose
  prepare-time conversion failed), so reprinting an old failed `.docx`
  job works. Stale converted PDFs are automatically regenerated when the
  source office file is newer. If the converted PDF doesn't match Word
  (wrong page count, shifted content), see
  [DOCX conversion inaccurate / extra pages](#docx-conversion-inaccurate--extra-pages)
  in Troubleshooting — missing fonts are the usual cause.
- **Other non-image, non-PDF attachments** are still sent to `lp` as-is;
  whether those print depends on your CUPS filters.
- **Attachment size limits for previews**: Discord (~25MB on most servers)
  and Gmail (~25MB) both cap attachment size. A combined PDF from many
  high-resolution photos could occasionally hit that; if it does, the bot
  will report it rather than silently failing to attach.
- **Email reply parsing**: `ai_classifier.classify_reply()` uses Gemini
  rather than keyword matching, so replies like "go ahead" or "nah, don't
  bother" work naturally — at the cost of one extra Gemini call per reply.
- **Cancelled jobs are terminal**: once cancelled (via either channel),
  further replies on that thread won't reopen it. If you want cancelled
  jobs to also be reprintable, add `STATUS_CANCELLED` to the retry check
  in `confirmation.ConfirmationManager.handle_approval`.

## Troubleshooting

### The bot says "printed successfully" but no paper comes out

`lp` accepting a job only means it entered the CUPS queue. The bot now
waits for the job to leave the queue and reports the printer state if it
gets stuck, but a job that "completes" with no output almost always means
the wrong driver is installed **or the Brother i386 filter can't run on
ARM** (see setup step 1 — `brlaser` does NOT support the DCP-J100 inkjet;
on a Pi, missing qemu wrapping causes silent "completed" jobs). Diagnose:

```bash
lpstat -v DCPJ100                      # which device URI the queue points at
lpstat -p DCPJ100 -l                   # printer state + last state message
lsusb | grep -i brother                # must show the printer when powered on
file /opt/brother/Printers/dcpj100/lpd/brdcpj100filter.real 2>/dev/null \
  || file /opt/brother/Printers/dcpj100/lpd/brdcpj100filter
# Expect: ELF 32-bit LSB executable, Intel i386
/opt/brother/Printers/dcpj100/lpd/brdcpj100filter; echo exit:$?
# exit 126 = Exec format error → install qemu wrap (setup step 1)
# "invalid option" / exit 2 = filter runs (qemu OK); try a real lp job
lp -d DCPJ100 /usr/share/cups/data/testprint
lpstat -W completed -o DCPJ100
sudo tail -50 /var/log/cups/error_log
```

If the test page "completes" without printing and `file` shows an i386
binary that exits 126, redo the qemu + `/opt/i386root` wrap in setup
step 1. If USB shows "Unplugged or turned off", power/cable first —
CUPS will disable the queue until the printer reappears
(`cupsenable DCPJ100`).

### Duplicate jobs for the same email thread / repeated approvals

Older versions could create a second job when a reply in the thread
carried the original attachment, and the two jobs could then approve
each other in a loop via the bot's own notification emails. This is
fixed (bot emails are tagged with an `X-Printbot` header and ignored;
one job per Gmail thread is enforced; quoted text is stripped before
reply classification), and duplicate jobs already in `state.json` are
skipped by the reply poller with a warning. To clean up for good: stop
the bot, open `state.json`, delete the newer duplicate entries under
`"jobs"` (the ones whose subject starts with `Re:`), and start the bot
again.

### DOCX conversion inaccurate / extra pages

**Symptom:** Word shows one page count (e.g. 2 pages) but the bot's PDF
preview or print shows more (e.g. 3 pages), with content spilling onto
the next page at the bottom.

**Cause:** LibreOffice lays out the document *before* exporting to PDF. If
a font used in the DOCX isn't installed on the Pi, LibreOffice substitutes
a different font with different character widths → line breaks shift →
pagination changes. This is the most common cause of inaccurate conversion;
PDF export settings cannot fix layout that was already calculated with the
wrong font. Many Word fonts (e.g. **Century Gothic**) are **not** included
in `ttf-mscorefonts-installer`.

**Solutions** (try in order):

1. **Install common Word font substitutes** (see the apt lines in setup
   step 1):
   - `fonts-crosextra-carlito` / `fonts-crosextra-caladea` → Calibri /
     Cambria substitutes
   - `fonts-liberation` → Arial / Times / Courier substitutes
   - `ttf-mscorefonts-installer` → Arial, Times New Roman, Verdana, etc.
   - Then refresh the font cache and delete any stale converted PDF:

   ```bash
   sudo fc-cache -f -v
   rm jobs/<message_id>/YourFile.pdf
   ```

2. **Identify which font the document uses** — check the font dropdown in
   Word, or inspect the DOCX on the Pi:

   ```bash
   unzip -p jobs/<message_id>/YourFile.docx word/fontTable.xml \
     | grep -oP 'w:ascii="\K[^"]+' | sort -u
   fc-match "Font Name Here"
   ```

   If `fc-match` returns a substitute (e.g. DejaVu Sans instead of
   Century Gothic), that font is missing.

3. **Copy a specific font from a Windows PC** — font files live in
   `C:\Windows\Fonts\` (e.g. `GOTHIC.TTF` for Century Gothic). Use the
   full path with `scp` (wildcards only work when run from the Fonts
   directory or with an explicit path):

   ```powershell
   ssh matt@<pi-ip> "mkdir -p ~/.local/share/fonts"
   scp C:\Windows\Fonts\GOTHIC*.TTF matt@<pi-ip>:~/.local/share/fonts/
   ```

   Then on the Pi:

   ```bash
   fc-cache -f -v
   fc-match "Century Gothic"
   ```

4. **Copy all Windows fonts** (optional, for maximum compatibility) —
   zip `.ttf`, `.ttc`, and `.otf` files from `C:\Windows\Fonts\`, copy
   the archive to the Pi, unzip into `~/.local/share/fonts/windows/`, and
   run `fc-cache -f -v`. Fine for personal use from a licensed Windows
   install; do not redistribute the font files.

5. **Workflow workarounds** when layout must match Word exactly:
   - Email a **PDF exported from Word** instead of the `.docx`
   - Or convert on a Windows machine with Word before sending

**Re-test after installing fonts:**

```bash
rm jobs/<message_id>/YourFile.pdf
cd ~/printbot && source venv/bin/activate
python3 -c "
from src.pdf_utils import office_to_pdf
print(office_to_pdf('jobs/<message_id>/YourFile.docx', '/tmp/test'))
"
```

## Recommendations

- **Start with a narrow Gmail search query.** The free Gemini tier has
  fairly low per-key, per-day request limits; scanning every attachment
  email in your inbox will burn through it fast. A query like
  `from:trusted@example.com has:attachment newer_than:1d` keeps volume low.
- **Get 2–3 Gemini keys from separate Google accounts**, not the same
  account's multiple projects — free-tier quota is often tied to the
  account/billing profile, not just the project.
- **Test printing manually first**: `lp -d Brother_DCP_J100 test.pdf`
  before wiring up the bot, so you know CUPS + driver + media names are
  correct.
- **Back up `credentials/token.json`** somewhere safe — regenerating it
  requires the browser flow again.
- **Leave `allow_non_owner_email_approval` off** unless you have a
  specific reason to let other people approve prints by replying — prefer
  the `approved_reply_senders` whitelist if you want to allow a few
  trusted people specifically.
- **Consider a stricter owner check on Discord** if the bot's channel is
  in a shared server — it already restricts button/command use to your
  user ID, but you may also want to make the channel private.
- **The Pi Zero 2 W is modest hardware.** Discord's gateway connection and
  the two polling loops (new emails + confirmation replies) are
  lightweight, but keep `poll_interval_seconds` at 60s or higher to avoid
  unnecessary load and API usage.
