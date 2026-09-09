# GBD Project Page

Official project page for "Generalized Biomedicine Discovery" (ECCV 2026).

## Local Development

To preview the page locally, you can use any static file server:

**Using Python:**
```bash
cd docs
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser.

**Using Node.js:**
```bash
cd docs
npx http-server -p 8000
```

**Using PHP:**
```bash
cd docs
php -S localhost:8000
```

## Structure

- `index.html` - Main project page
- `styles.css` - Styling and animations
- `script.js` - Interactive features
- `assets/` - Images and figures
- `ECCV2026___Generalized_Biomedicine_Discovery__Camera_Ready_.pdf` - Paper PDF

## Deployment

This page can be deployed to GitHub Pages, Netlify, Vercel, or any static hosting service.

For GitHub Pages:
1. Push the `docs` folder to your repository
2. Enable GitHub Pages in repository settings
3. Set source to the `docs` folder
