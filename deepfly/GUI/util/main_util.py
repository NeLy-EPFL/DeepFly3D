def button_set_width(btn, text=" ", margin=20):
    width = btn.fontMetrics().boundingRect(text).width() + 7 + margin
    btn.setMaximumWidth(width)