import gradio as gr


TITLE=""

TOP = """\
<div class="top">
        <div class="top-container">
                <img class="logo" width="140" height="37" src="https://kuae-playground.mthreads.com/image/logo@2x.png">
        </div>
</div>"""

FOOTER = '''\
<div class="footer">
    <span>Copyright © 2024-2025 摩尔线程版权所有 京公网安备 11010802036174号 京ICP证2020041674号</span>
</div>'''
js_change_title = '''\
window.onload = function() {
    document.title = "''' + TITLE + '''";
}'''
HEADER = TOP + "<h1>" + TITLE + "</h1><p>"



CSS='''body {
    margin: 0;
    background: #F8F8F8;
    font-size: 22px;
    color: #666666;
}
p {
    font-size: 16px;
}
.top {
    left: 0;
    top: 0;
    height: 3.83%;
    opacity: 1;
    justify-content: center;
    background: white;
}
.top-container {
    /* 原样式 */
    margin: 0 auto;
    max-width: 1500px;
    padding: 10px;
    display: flex;
    padding: 16px 0;
    overflow: hidden;
}
.logo {
    margin: 0;
    padding: 0 20px;
}
h2 {
    position: relative;
    margin: 0;
    font-size: 21px;
    font-weight: normal;
    line-height: 20px;
    letter-spacing: 0;
    padding: 5px 20px;
    color: #666666;
}
.top-container>h2:before {
    background: #dcdfe6;
    /* 设置背景颜色为浅灰色 */
    content: "";
    /* 伪元素的内容，这里为空，意味着不会显示任何文本 */
    height: 16px;
    /* 设置伪元素的高度为 16px */
    left: 0;
    /* 设置伪元素的左边距为 0，即与其定位的父元素（这里是 h2）的左边对齐 */
    position: absolute;
    /* 设置位置为绝对定位，从而允许我们根据父元素进行准确放置 */
    top: 50%;
    /* 设置伪元素的上边距为 50%，这样伪元素的顶部将对齐到父元素的中间 */
    transform: translateY(-50%);
    /* 将伪元素向上移动自身高度的一半，即使其完全居中于父元素 */
    width: 1px;
    /* 设置伪元素的宽度为 1px，表现为一个细线 */
}
h1 {
  text-align: center;
  display: block;
}

.footer {
    padding: 20px;
    text-align: center;
    font-size: 16px;
}

.footer .logo {
    display: inline-block;
    /* 内联块元素使其与文本对齐 */
    margin-right: 10px;
    /* 右边距 */
}

.footer a {
    color: #666666;
    text-decoration: none;
}
footer {
        visible;
}'''


class Blocks(gr.Blocks):  
    def __init__(self, **kwargs):  
        super().__init__(css=CSS, js=js_change_title)  

    def __enter__(self):  
        r = super().__enter__()  
        gr.HTML(HEADER)
        return r

    def __exit__(self, exc_type, exc_value, traceback):  
        gr.HTML(FOOTER)
        return super().__exit__(exc_type, exc_value, traceback)  
