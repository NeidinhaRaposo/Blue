function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const btnIcon = document.querySelector('.theme-toggle i');
    if (document.body.classList.contains('dark-mode')) {
        btnIcon.classList.remove('fa-moon');
        btnIcon.classList.add('fa-sun');
    } else {
        btnIcon.classList.remove('fa-sun');
        btnIcon.classList.add('fa-moon');
    }
}    