document.addEventListener("DOMContentLoaded", function() {
    // Find all abbr tags with a title attribute
    var abbrs = document.querySelectorAll('abbr[title]');
    abbrs.forEach(function(abbr) {
        // Move the title to data-custom-tooltip to suppress the native browser tooltip
        abbr.setAttribute('data-custom-tooltip', abbr.getAttribute('title'));
        abbr.removeAttribute('title');
    });
});
