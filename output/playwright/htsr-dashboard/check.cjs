const { chromium } = require('/thayerfs/home/f007h3r/.npm/_npx/f0a362733743bae2/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    colorScheme: 'dark',
  });
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('http://127.0.0.1:8501', { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(12000);
  for (const tab of ['Prediction', 'Lineage', 'Evidence', 'Data']) {
    await page.getByText(tab, { exact: true }).first().click();
    await page.waitForTimeout(5000);
    await page.screenshot({ path: `dashboard-${tab.toLowerCase()}.png`, fullPage: false });
    const body = await page.locator('body').innerText();
    console.log(`TAB=${tab} HAS_EXCEPTION=${/Traceback|Exception|Error running app/.test(body)}`);
  }
  console.log(`CONSOLE_ERRORS=${errors.length}`);
  if (errors.length) console.log(errors.slice(0, 8).join('\n'));
  await browser.close();
})();
