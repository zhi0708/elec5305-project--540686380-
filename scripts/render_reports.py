"""Typeset the measured report and revised proposal; no new measurements."""
import argparse,html,re,shutil
from pathlib import Path
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.font_manager as font_manager

# Prefer the thesis font when installed, with a metrically compatible fallback.
for name, weight, style in [
    ('ProjectSerif', 'normal', 'normal'),
    ('ProjectSerif-Bold', 'bold', 'normal'),
    ('ProjectSerif-Italic', 'normal', 'italic'),
    ('ProjectSerif-BoldItalic', 'bold', 'italic'),
]:
    properties = font_manager.FontProperties(
        family=['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        weight=weight, style=style,
    )
    path = font_manager.findfont(properties)
    pdfmetrics.registerFont(TTFont(name, path))
pdfmetrics.registerFontFamily(
    'ProjectSerif', normal='ProjectSerif', bold='ProjectSerif-Bold',
    italic='ProjectSerif-Italic', boldItalic='ProjectSerif-BoldItalic',
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
ROOT=Path(__file__).resolve().parents[1]
WIDTH,HEIGHT=A4
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle',fontName='ProjectSerif-Bold',fontSize=21,leading=24,
                          textColor=colors.black,spaceAfter=18))
styles.add(ParagraphStyle(name='ReportH1',fontName='ProjectSerif-Bold',fontSize=16,leading=18,
                          textColor=colors.black,spaceBefore=16,spaceAfter=9,keepWithNext=True))
styles.add(ParagraphStyle(name='ReportH2',fontName='ProjectSerif-Bold',fontSize=12.5,leading=15,
                          textColor=colors.black,spaceBefore=12,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='ReportBody',fontName='ProjectSerif',fontSize=11,leading=12.7,
                          spaceAfter=5,splitLongWords=True))
styles.add(ParagraphStyle(name='ReportCell',fontName='ProjectSerif',fontSize=9.5,leading=11))
styles.add(ParagraphStyle(name='ReportCaption',fontName='ProjectSerif',fontSize=9.5,leading=11.3,
                          textColor=colors.black,spaceAfter=12))


def text(value):
    value=value.replace('−','-').replace('–','-').replace('—','-')
    pieces = re.split(r'(\[[^\]]+\]\(https://[^)]+\))', value)
    for i, piece in enumerate(pieces):
        match = re.fullmatch(r'\[([^\]]+)\]\((https://[^)]+)\)', piece)
        if match:
            label, url = match.groups()
            pieces[i] = f'<link href="{html.escape(url, quote=True)}" color="#000000"><u>{html.escape(label)}</u></link>'
        else:
            pieces[i] = html.escape(piece)
    value = ''.join(pieces)
    value=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',value)
    return value


FIGURES = {
    '01_detection.png': ('Figure 1. Detection by strength and acoustic location',
                         'Original benchmark. Precision, recall and F1 by location and level.'),
    '02_tradeoff.png': ('Figure 2. Restoration versus alteration outside the event',
                        'Original benchmark. Lower error on both axes is desirable. True-mask interpolation is diagnostic.'),
    '06_extended_miss.png': ('Figure 3. Matched weak and strong keyboard insertions',
                             'Same held-out recording and insertion interval. Only the stronger event crosses the frozen threshold.'),
}


def figure(name):
    path = ROOT/'docs/assets/figures'/name
    title, caption = FIGURES[name]
    artwork = Image(str(path))
    scale = min((WIDTH-114)/artwork.imageWidth, 265/artwork.imageHeight)
    artwork.drawWidth = artwork.imageWidth*scale
    artwork.drawHeight = artwork.imageHeight*scale
    return KeepTogether([artwork, Spacer(1, 5),
                         Paragraph(f'<b>{title}.</b> {caption}', styles['ReportCaption'])])


class ReportDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'ReportH1':
            self.notify('TOCEntry', (0, flowable.getPlainText(), self.page))


def markdown_elements(source, with_contents=False):
    lines=source.splitlines();items=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('|'):
            data=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',s) for s in cells):
                    data.append([Paragraph(text(s),styles['ReportCell']) for s in cells])
                i+=1
            n=len(data[0]);width=WIDTH-114
            widths=[115]+[(width-115)/(n-1)]*(n-1)
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f2f2f2')),
                                  ('LINEBELOW',(0,0),(-1,0),.7,colors.black),
                                  ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#bbbbbb')),
                                  ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),
                                  ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            items.extend([t,Spacer(1,10)]);continue
        if line.startswith('!['):
            items.append(figure(Path(line.rsplit('](', 1)[1].removesuffix(')')).name))
        elif line == '## 1. Research question and motivation' and with_contents:
            contents = TableOfContents()
            contents.levelStyles = [styles['ReportBody']]
            items.extend([Paragraph('Contents', styles['ReportH2']), contents, PageBreak(),
                          Paragraph(text(line[3:]), styles['ReportH1'])])
        elif line == '## References' and with_contents:
            items.extend([PageBreak(), Paragraph('References', styles['ReportH1'])])
        elif line.startswith('# '):items.append(Paragraph(text(line[2:]),styles['ReportTitle']))
        elif line.startswith('## '):items.append(Paragraph(text(line[3:]),styles['ReportH1']))
        elif line.startswith('### '):items.append(Paragraph(text(line[4:]),styles['ReportH2']))
        elif re.match(r'^\d+\. ',line):items.append(Paragraph(text(line),styles['ReportBody']))
        else:
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')) and not re.match(r'^\d+\. ',lines[i]):
                paragraph.append(lines[i].strip());i+=1
            items.append(Paragraph(text(' '.join(paragraph)),styles['ReportBody']));continue
        i+=1
    return items


def footer(canvas,doc):
    canvas.setFont('ProjectSerif', 11)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(WIDTH/2, 27, str(doc.page))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', nargs='?', choices=['final_report', 'revised_proposal'])
    chosen = parser.parse_args().report
    for basename in ([chosen] if chosen else ['final_report', 'revised_proposal']):
        source=(ROOT/'report'/f'{basename}.md').read_text()
        elements=markdown_elements(source, with_contents=basename=='final_report')
        output=ROOT/'report'/f'{basename}.pdf'
        doc=ReportDocument(str(output),pagesize=A4,rightMargin=51,leftMargin=51,topMargin=51,
                           bottomMargin=51,title=source.splitlines()[0][2:],author='Leona Wang')
        doc.multiBuild(elements,onFirstPage=footer,onLaterPages=footer)
        shutil.copyfile(output,ROOT/'docs'/output.name)
        print(output)


if __name__=='__main__':main()
