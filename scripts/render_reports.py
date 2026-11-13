"""Typeset the measured report and revised proposal; no new measurements."""
import argparse,html,io,re,shutil
from pathlib import Path
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.font_manager as font_manager

# Embed portable fonts instead of relying on viewer-specific Base-14 substitutes.
for name, weight in [("ProjectSans", "normal"), ("ProjectSans-Bold", "bold")]:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))
    pdfmetrics.registerFont(TTFont(name, path))
pdfmetrics.registerFontFamily("ProjectSans", normal="ProjectSans", bold="ProjectSans-Bold", italic="ProjectSans", boldItalic="ProjectSans-Bold")
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
ROOT=Path(__file__).resolve().parents[1]
WIDTH,HEIGHT=A4
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle',fontName='ProjectSans-Bold',fontSize=23,leading=28,
                          textColor=colors.HexColor('#183440'),spaceAfter=18))
styles.add(ParagraphStyle(name='ReportH1',fontName='ProjectSans-Bold',fontSize=14,leading=18,
                          textColor=colors.HexColor('#197b80'),spaceBefore=16,spaceAfter=9,keepWithNext=True))
styles.add(ParagraphStyle(name='ReportH2',fontName='ProjectSans-Bold',fontSize=11.5,leading=15,
                          textColor=colors.HexColor('#183440'),spaceBefore=12,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='ReportBody',fontName='ProjectSans',fontSize=10,leading=14,
                          spaceAfter=8,splitLongWords=True))
styles.add(ParagraphStyle(name='ReportCell',fontName='ProjectSans',fontSize=7.6,leading=10))
styles.add(ParagraphStyle(name='ReportCaption',fontName='ProjectSans',fontSize=9,leading=12,
                          textColor=colors.HexColor('#52656b'),spaceAfter=12))


def text(value):
    value=value.replace('−','-').replace('–','-').replace('—','-')
    value=html.escape(value)
    value=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',value)
    return value


def markdown_elements(source):
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
            n=len(data[0]);width=WIDTH-90
            widths=[115]+[(width-115)/(n-1)]*(n-1)
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f0ed')),
                                  ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#197b80')),
                                  ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d4ddda')),
                                  ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),
                                  ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            items.extend([t,Spacer(1,10)]);continue
        if line.startswith('# '):items.append(Paragraph(text(line[2:]),styles['ReportTitle']))
        elif line.startswith('## '):items.append(Paragraph(text(line[3:]),styles['ReportH1']))
        elif line.startswith('### '):items.append(Paragraph(text(line[4:]),styles['ReportH2']))
        elif re.match(r'^\d+\. ',line):items.append(Paragraph(text(line),styles['ReportBody']))
        else:
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|')) and not re.match(r'^\d+\. ',lines[i]):
                paragraph.append(lines[i].strip());i+=1
            items.append(Paragraph(text(' '.join(paragraph)),styles['ReportBody']));continue
        i+=1
    return items


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#cedad6'));canvas.line(45,36,WIDTH-45,36)
    canvas.setFont('ProjectSans',8);canvas.setFillColor(colors.HexColor('#617279'))
    canvas.drawString(45,24,'Leona Wang | SID 540686380 | ELEC5305')
    canvas.drawRightString(WIDTH-45,24,str(doc.page))


def main():
    from PIL import Image as PILImage
    parser = argparse.ArgumentParser()
    parser.add_argument('report', nargs='?', choices=['final_report', 'revised_proposal'])
    chosen = parser.parse_args().report
    for basename in ([chosen] if chosen else ['final_report', 'revised_proposal']):
        source=(ROOT/'report'/f'{basename}.md').read_text()
        elements=markdown_elements(source)
        if basename=='final_report':
            for name,title,caption in [
                ('01_detection.png','Figure 1. Detection by strength and acoustic location','Held-out event metrics; all event templates pooled within each overlap/strength condition.'),
                ('02_tradeoff.png','Figure 2. Event restoration versus collateral alteration','Lower error on both axes is desirable. True-mask interpolation is diagnostic only.'),
                ('03_overlap_waveforms.png','Figure 3. Identical event at different locations','A fixed 20 ms event waveform and absolute amplitude are used for all three insertions.'),
                ('04_clean_false_positive.png','Figure 4. False detection on clean reference speech','The high normalised flux peak triggers an event despite low absolute energy. No phoneme label is available.'),
                ('05_quality_runtime.png','Figure 5. Restoration quality and CPU runtime','Processing is timed on the same CPU. Model loading and warm-up are excluded; no causal latency is inferred.'),
                ('06_extended_miss.png','Figure 6. Matched weak and strong keyboard insertions','Same unseen recording, event template and insertion interval. The score crosses the threshold for the stronger event only.')]:
                path=ROOT/'docs/assets/figures'/name
                with PILImage.open(path) as artwork:
                    artwork = artwork.convert('RGB')
                    artwork.thumbnail((1400, 1400))
                    iw, ih = artwork.size
                    encoded = io.BytesIO()
                    artwork.save(encoded, format='JPEG', quality=86, optimize=True)
                    encoded.seek(0)
                scale=min((WIDTH-90)/iw,(HEIGHT-160)/ih)
                elements.extend([PageBreak(),Paragraph(title,styles['ReportH1']),
                                 Image(encoded,width=iw*scale,height=ih*scale),
                                 Paragraph(caption,styles['ReportCaption'])])
        output=ROOT/'report'/f'{basename}.pdf'
        doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=45,
                              bottomMargin=48,title=source.splitlines()[0][2:],author='Leona Wang')
        doc.build(elements,onFirstPage=footer,onLaterPages=footer)
        shutil.copyfile(output,ROOT/'docs'/output.name)
        print(output)


if __name__=='__main__':main()
