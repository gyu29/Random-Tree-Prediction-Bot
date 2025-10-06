const { contextBridge } = require('electron')
contextBridge.exposeInMainWorld('electronAPI', {
  ping: () => console.log("Ping from preload.js")
})
