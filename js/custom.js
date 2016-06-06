$('#nav').affix({
    offset: {
        top: $('#nav').offset().top
    }
});

$('#nav').affix({
    offset: {
        bottom: ($('footer').outerHeight(true) +
                $('.application').outerHeight(true)) +
            40
    }
});