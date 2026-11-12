# Build Stage
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies (use npm ci when a lockfile is present for reproducible builds)
COPY package*.json ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# Copy source
COPY . .

# Empty API base => the app calls /api and /output on its own origin, which
# nginx proxies to the backend container. Override at build time if needed.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE

RUN npm run build

# Production Stage with Nginx
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
# The nginx image fills ${FRAME_ANCESTORS} into this template when it starts.
COPY templates/default.conf.template /etc/nginx/templates/default.conf.template
ENV FRAME_ANCESTORS="'self'"

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
