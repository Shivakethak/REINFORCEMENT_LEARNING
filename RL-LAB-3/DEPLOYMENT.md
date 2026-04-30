# Deployment Guide - RL Agent Evaluation Dashboard

## 🚀 Quick Deployment Options

### 1. **Streamlit Cloud (Recommended - Free)**
1. Push your code to GitHub/GitLab
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `RL-LAB-3` folder
5. Main file: `app.py`
6. Click **Deploy**

### 2. **Railway (Easy - $5/month)**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### 3. **Docker (Self-hosted)**
```bash
# Build and run with Docker
docker build -t rl-dashboard .
docker run -p 8501:8501 rl-dashboard

# Or with docker-compose
docker-compose up -d
```

### 4. **Heroku (Free tier available)**
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

### 5. **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 📋 Requirements for Deployment

- **Python**: 3.11+
- **Memory**: Minimum 1GB RAM
- **Storage**: 500MB+
- **Port**: 8501 (configurable)

## 🔧 Configuration Files Created

- `Dockerfile` - Container configuration
- `docker-compose.yml` - Multi-container setup
- `Procfile` - Heroku deployment
- `railway.toml` - Railway configuration
- `.dockerignore` - Optimize Docker builds

## 🌐 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | 8501 | Server port |
| `STREAMLIT_SERVER_ADDRESS` | 0.0.0.0 | Server address |

## 🔍 Health Check

Your app includes a health check endpoint: `/_stcore/health`

## 📊 Performance Tips

1. **Enable caching** in Streamlit for faster loads
2. **Use CDN** for static assets if deploying at scale
3. **Monitor memory usage** - RL training can be intensive
4. **Set timeouts** for long training sessions

## 🐛 Troubleshooting

### Common Issues:
- **Memory errors**: Reduce batch size or use smaller environments
- **Slow loading**: Add `@st.cache_data` to expensive functions
- **Port conflicts**: Change port in config or use different service port

### Docker Issues:
```bash
# Check logs
docker logs <container-id>

# Debug mode
docker run -it --entrypoint /bin/bash rl-dashboard
```

## 📞 Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Project Issues**: Check GitHub Issues
- **Deployment Help**: Refer to platform-specific docs
