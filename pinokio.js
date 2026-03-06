module.exports = {
  version: "2.0",
  title: "MilimoVideo-Lite",
  description: "MilimoVideo with automatic low-VRAM backend routing and sandboxed runtime.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const installed = info.exists("sandbox/venv") && info.exists("sandbox/workspace/milimovideo");
    const running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
    };

    if (running.install) {
      return [
        {
          default: true,
          icon: "fa-solid fa-plug",
          text: "Installing",
          href: "install.js",
        },
      ];
    }

    if (!installed) {
      return [
        {
          default: true,
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        },
        {
          icon: "fa-solid fa-terminal",
          text: "Update",
          href: "update.js",
        },
      ];
    }

    if (running.start) {
      return [
        {
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open UI",
          href: "http://127.0.0.1:5173/",
        },
        {
          icon: "fa-solid fa-database",
          text: "Backend API",
          href: "http://127.0.0.1:8000/docs",
        },
        {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        },
      ];
    }

    if (running.update) {
      return [
        {
          default: true,
          icon: "fa-solid fa-arrows-rotate",
          text: "Updating",
          href: "update.js",
        },
      ];
    }

    return [
      {
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.js",
      },
      {
        icon: "fa-solid fa-plug",
        text: "Reinstall",
        href: "install.js",
      },
      {
        icon: "fa-solid fa-arrows-rotate",
        text: "Update",
        href: "update.js",
      },
      {
        icon: "fa-solid fa-folder-open",
        text: "Open Sandbox",
        href: "sandbox",
      },
      {
        icon: "fa-solid fa-file-lines",
        text: "Backend Log",
        href: "sandbox/workspace/milimovideo/backend/server.log",
      },
      {
        icon: "fa-solid fa-sliders",
        text: "Lite Config",
        href: "sandbox/config.json",
      },
    ];
  },
};
