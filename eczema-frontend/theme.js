(function () {
    const stored = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = stored ? stored === 'dark' : prefersDark;
    if (isDark) document.documentElement.classList.add('dark');
})();

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeToggleIcon();
}

function updateThemeToggleIcon() {
    const isDark = document.documentElement.classList.contains('dark');
    document.querySelectorAll('[data-theme-icon]').forEach(function (el) {
        el.textContent = isDark ? '☀️' : '\u{1F319}';
    });
}

document.addEventListener('DOMContentLoaded', updateThemeToggleIcon);
