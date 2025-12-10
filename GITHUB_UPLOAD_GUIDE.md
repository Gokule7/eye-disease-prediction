# GitHub Upload Instructions

## Step-by-Step Guide to Upload Your Project to GitHub

### Option 1: Using GitHub Desktop (Easiest)

1. **Download GitHub Desktop**:
   - Go to: https://desktop.github.com/
   - Download and install

2. **Sign in to GitHub**:
   - Open GitHub Desktop
   - Sign in with your GitHub account
   - If you don't have one, create at: https://github.com/join

3. **Create Repository**:
   - Click "File" → "Add Local Repository"
   - Navigate to: `D:\EyeProject\EyeProject`
   - Click "create a repository" link
   - Name: `eye-disease-gradcam`
   - Description: "Eye Disease Classification with Grad-CAM Visualization"
   - Click "Create Repository"

4. **Publish to GitHub**:
   - Click "Publish repository" button
   - Uncheck "Keep this code private" (or keep checked for private repo)
   - Click "Publish repository"

5. **Done!** Your project is now on GitHub 🎉

---

### Option 2: Using Git Command Line

1. **Install Git**:
   - Download from: https://git-scm.com/downloads
   - Install with default settings

2. **Open PowerShell** in your project folder:
   ```powershell
   cd D:\EyeProject\EyeProject
   ```

3. **Initialize Git repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Eye Disease Classification with Grad-CAM"
   ```

4. **Create GitHub repository**:
   - Go to: https://github.com/new
   - Repository name: `eye-disease-gradcam`
   - Description: "Eye Disease Classification with Grad-CAM Visualization"
   - Choose Public or Private
   - Do NOT initialize with README (we already have one)
   - Click "Create repository"

5. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/eye-disease-gradcam.git
   git branch -M main
   git push -u origin main
   ```
   Replace `YOUR_USERNAME` with your actual GitHub username

---

### Option 3: Using Visual Studio Code (If you have it)

1. **Open project in VS Code**:
   - File → Open Folder → Select `D:\EyeProject\EyeProject`

2. **Initialize Git**:
   - Click Source Control icon (left sidebar)
   - Click "Initialize Repository"

3. **Commit changes**:
   - Stage all files (click + icon)
   - Write message: "Initial commit: Eye Disease Classification with Grad-CAM"
   - Click ✓ (checkmark) to commit

4. **Publish to GitHub**:
   - Click "Publish to GitHub" button
   - Choose repository name: `eye-disease-gradcam`
   - Select Public or Private
   - Click "Publish"

---

## Important Notes Before Uploading

### ✅ Already Prepared:
- README.md (project documentation)
- .gitignore (excludes large files and datasets)
- LICENSE (MIT License)
- requirements.txt (Python dependencies)
- Grad-CAM results (5 visualization images)
- Composite figure
- All Python scripts

### ⚠️ Files Excluded (via .gitignore):
- ODIR-5K dataset (too large for GitHub)
- Trained model files (.h5, .keras)
- Preprocessed images
- __pycache__ and temporary files

### 📝 After Uploading:

1. **Update README.md** with:
   - Your actual GitHub username
   - Your email address
   - Any specific instructions

2. **Add dataset instructions**:
   Users will need to download ODIR-5K separately from:
   https://odir2019.grand-challenge.org/

3. **Share your repository**:
   - Repository URL will be: `https://github.com/YOUR_USERNAME/eye-disease-gradcam`
   - Share with collaborators
   - Add topics/tags for discoverability

---

## Useful Git Commands (After Initial Setup)

### Check status:
```bash
git status
```

### Add new changes:
```bash
git add .
git commit -m "Your commit message"
git push
```

### Update from GitHub:
```bash
git pull
```

### View history:
```bash
git log
```

---

## GitHub Repository Settings (After Upload)

### Recommended Actions:

1. **Add Topics** (on GitHub website):
   - machine-learning
   - deep-learning
   - medical-imaging
   - grad-cam
   - eye-disease
   - tensorflow
   - computer-vision
   - explainable-ai

2. **Add Description**:
   "Eye Disease Classification with Grad-CAM Visualization for Interpretable AI in Medical Imaging"

3. **Enable GitHub Pages** (optional):
   - Settings → Pages
   - Display your README as a website

4. **Add Collaborators** (if working with a team):
   - Settings → Collaborators
   - Invite team members

---

## Troubleshooting

### Large files error:
- Already handled by .gitignore
- If error occurs, the dataset/models are too large
- Solution: Files are already excluded

### Authentication required:
- Use GitHub Desktop (easiest)
- Or generate Personal Access Token:
  - GitHub → Settings → Developer settings → Personal access tokens
  - Use token as password

### Push rejected:
```bash
git pull origin main
git push origin main
```

---

## Need Help?

- GitHub Docs: https://docs.github.com/
- Git Tutorial: https://git-scm.com/docs/gittutorial
- GitHub Desktop Guide: https://docs.github.com/desktop

---

**Your project is ready to upload! Choose the option that works best for you.** 🚀
