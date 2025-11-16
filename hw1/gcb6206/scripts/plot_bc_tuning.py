import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 튜닝 결과 데이터
steps = np.array([10, 100, 500, 1000])
eval_returns = np.array([-543.29, 330.30, 3376.13, 3721.04])

# 2. 그래프 설정
plt.figure(figsize=(8, 5))
plt.style.use('ggplot')  # 그래프 스타일 설정 (선택 사항)

# 3. 꺾은선 그래프 플롯
# 'o-'는 마커(점)와 선을 함께 표시합니다.
plt.plot(steps, eval_returns, marker='o', linestyle='-', color='tab:blue', linewidth=2)

# 4. 축 및 레이블 설정
plt.title('BC Performance vs. Training Steps (HalfCheetah-v4)', fontsize=14)
plt.xlabel('Number of Agent Training Steps per Iteration', fontsize=12)
plt.ylabel('Evaluation Average Return', fontsize=12)

# X축 눈금 설정 (튜닝 값에 맞춰 명확하게 표시)
plt.xticks(steps, [f'{s}' for s in steps])

# 그리드 추가 (ggplot 스타일에서는 기본)
plt.grid(True, linestyle='--', alpha=0.6)

# y=0을 나타내는 수평선 추가
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)

# 5. 파일 저장 (PDF 형식)
output_filename = 'bc_tuning_results.pdf'

# 현재 디렉토리의 'assets' 폴더에 저장 (없으면 'hw1' 디렉토리에 저장됨)
try:
    output_path = os.path.join(os.getcwd(), 'assets', output_filename)
    os.makedirs(os.path.join(os.getcwd(), 'assets'), exist_ok=True)
except:
    output_path = os.path.join(os.getcwd(), output_filename)

plt.savefig(output_path, bbox_inches='tight')
print(f"✅ 그래프가 다음 경로에 저장되었습니다: {output_path}")

# 그래프 출력 (선택 사항, 로컬에서만 작동)
# plt.show()