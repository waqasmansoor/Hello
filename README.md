<h1>For Production</h1>
<p>Goto the react app directory</p>

```bash
cd web_app/voice-recorder-app
```

<p>Install <b>npm</b> packages</p>

```bash
npm install .
```

<p>Create <b>npm build</b></p>

```bash
npm run build
```

<p>Run app with <b>pm2</b></p>

```bash
pm2 start npm --name vr -- run dev -- --host 0.0.0.0 --port 3000

```
