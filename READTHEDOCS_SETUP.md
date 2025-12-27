# Read the Docs Setup Guide for toksum

## 📚 Overview

This guide will help you set up toksum documentation on Read the Docs for professional hosting.

## ✅ Prerequisites Completed

I've already set up all the necessary files:

- ✅ `.readthedocs.yaml` - Read the Docs configuration
- ✅ `docs/requirements.txt` - Documentation dependencies
- ✅ `docs/conf.py` - Enhanced Sphinx configuration
- ✅ Complete documentation structure with examples
- ✅ Version 0.7.0 with comprehensive docs

## 🚀 Read the Docs Setup Steps

### 1. Push Your Documentation Branch

Make sure your `docs/sphinx-documentation` branch is pushed to GitHub:

```bash
# Commit any remaining changes
git add .readthedocs.yaml READTHEDOCS_SETUP.md
git commit -m "Add Read the Docs configuration"
git push origin docs/sphinx-documentation
```

### 2. Create Pull Request and Merge

1. Go to your GitHub repository
2. Create a Pull Request from `docs/sphinx-documentation` to `main`
3. Merge the PR (this will trigger Read the Docs build)

### 3. Set Up Read the Docs Project

1. **Go to Read the Docs**: https://readthedocs.org/
2. **Sign in** with your GitHub account
3. **Import a Project**:
   - Click "Import a Project"
   - Select your `kactlabs/toksum` repository
   - Click "Next"

### 4. Configure Project Settings

**Basic Settings:**
- **Name**: `toksum`
- **Repository URL**: `https://github.com/kactlabs/toksum`
- **Default branch**: `main`
- **Language**: `English`
- **Programming Language**: `Python`

**Advanced Settings:**
- **Python configuration file**: `docs/conf.py`
- **Requirements file**: `docs/requirements.txt`
- **Python interpreter**: `CPython 3.11`

### 5. Build Configuration

Read the Docs will automatically use our `.readthedocs.yaml` configuration:

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
   configuration: docs/conf.py
python:
   install:
   - requirements: docs/requirements.txt
   - method: pip
     path: .
```

### 6. Trigger First Build

1. Click **"Build version"** in your Read the Docs dashboard
2. Monitor the build process
3. Check for any errors in the build log

## 📖 Expected Documentation Structure

Your Read the Docs site will include:

- **Home Page**: Overview with quick start guide
- **API Reference**: Complete API documentation
  - Core module (TokenCounter, functions)
  - CLI module (command-line interface)
  - Exceptions (error handling)
- **Examples**: Comprehensive usage examples
- **Changelog**: Version history

## 🔗 URLs After Setup

Once configured, your documentation will be available at:

- **Main URL**: `https://toksum.readthedocs.io/`
- **Latest**: `https://toksum.readthedocs.io/en/latest/`
- **Stable**: `https://toksum.readthedocs.io/en/stable/`

## 🛠 Troubleshooting

### Common Issues:

**Build Fails - Missing Dependencies:**
```bash
# Our docs/requirements.txt includes all needed dependencies:
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
tiktoken>=0.5.0  # Core dependency for OpenAI models
```

**Import Errors:**
- Our `.readthedocs.yaml` installs the package itself: `pip install .`
- This ensures all imports work correctly

**Theme Issues:**
- We're using `sphinx_rtd_theme` which is the standard Read the Docs theme
- All theme options are configured in `docs/conf.py`

### Build Logs:
Check the build logs in Read the Docs dashboard for specific errors.

## 🎯 Post-Setup Tasks

### 1. Set Up Webhooks (Automatic)
Read the Docs automatically sets up GitHub webhooks for:
- ✅ Push to main branch → rebuild docs
- ✅ New releases → rebuild docs
- ✅ Pull requests → preview builds

### 2. Configure Versions
- **latest**: Always points to main branch
- **stable**: Points to latest release tag
- **v0.7.0**: Specific version documentation

### 3. Custom Domain (Optional)
You can set up a custom domain like `docs.toksum.com` in the Read the Docs settings.

## 📊 Features You'll Get

✅ **Professional Documentation**: Clean, searchable, mobile-responsive
✅ **Automatic Builds**: Updates on every push to main
✅ **Version Management**: Multiple versions (latest, stable, tagged)
✅ **Search Functionality**: Full-text search across all docs
✅ **PDF Downloads**: Automatic PDF generation
✅ **Analytics**: Built-in traffic analytics
✅ **SSL Certificate**: HTTPS by default
✅ **CDN**: Fast global content delivery

## 🔄 Maintenance

### Updating Documentation:
1. Update docstrings in Python files
2. Push to main branch
3. Read the Docs automatically rebuilds

### New Releases:
1. Create a new git tag: `git tag v0.7.0`
2. Push tag: `git push origin v0.7.0`
3. Read the Docs creates version-specific docs

## 📞 Support

If you encounter issues:
- Check Read the Docs build logs
- Review our `.readthedocs.yaml` configuration
- Ensure all dependencies are in `docs/requirements.txt`
- Test local build: `python build_docs.py build`

Your toksum documentation is now ready for professional hosting on Read the Docs! 🎉