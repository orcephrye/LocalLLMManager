document.addEventListener('DOMContentLoaded', () => {
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeToggleIcon = themeToggleBtn ? themeToggleBtn.querySelector('.theme-toggle-icon') : null;
    
    // Apply initial theme from localStorage or default to system preference
    const getSavedTheme = () => {
        const localTheme = localStorage.getItem('theme');
        if (localTheme) return localTheme;
        
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        return systemPrefersDark ? 'dark' : 'light';
    };

    const savedTheme = getSavedTheme();
    document.documentElement.setAttribute('data-theme', savedTheme);
    if (themeToggleIcon) {
        themeToggleIcon.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            if (themeToggleIcon) {
                themeToggleIcon.textContent = newTheme === 'dark' ? '☀️' : '🌙';
            }
        });
    }
});
