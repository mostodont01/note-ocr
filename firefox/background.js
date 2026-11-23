// note-OCR: toolbar button -> selection box on the page -> capture -> native host.
const HOST = "notocr";

browser.browserAction.onClicked.addListener(async (tab) => {
  try {
    await browser.tabs.executeScript(tab.id, { file: "overlay.js" });
  } catch (e) {
    // Pages where scripts can't run (Firefox's PDF viewer, about: pages):
    // send the whole visible tab; crop it in the app.
    await recognise(tab.windowId, null, 1, "open");
  }
});

browser.runtime.onMessage.addListener((msg, sender) => {
  if (msg.type === "capture") {
    return recognise(sender.tab.windowId, msg.rect, msg.dpr, msg.mode);
  }
});

async function recognise(windowId, rect, dpr, mode) {
  const image = await browser.tabs.captureVisibleTab(windowId, { format: "png" });
  try {
    return await browser.runtime.sendNativeMessage(HOST, {
      action: "recognise", image, rect, dpr, mode,
    });
  } catch (e) {
    return { ok: false, error: "note-OCR helper not reachable (run `notocr install-firefox`): " + e.message };
  }
}
