---
title: EU Explorer       # the display name of your Space
emoji: 🤖                         # a single emoji that represents your app
colorFrom: purple                 # one of: red, yellow, green, blue, indigo, purple, pink, gray
colorTo: indigo                   # another one from the same list, for the gradient
sdk: docker                       # since you’re using a Dockerfile
# sdk_version is only for Gradio/Streamlit; you can omit it for docker
app_port: 7860                    # (optional) if your Docker container listens on a port other than 7860
app_file: ./backend/main.py                  # the entry-point script in your repo
pinned: false                     # whether to pin this Space in your profile
---
