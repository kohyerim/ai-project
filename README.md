## 제주대학교 인공지능 기말 프로젝트

- 주제 : 중고차 시세 예측

- 데이터셋 출처 및 참고 사이트 : https://www.kaggle.com/avikasliwal/used-cars-price-prediction

  --------

  ### mllib - prediction_util

  https://github.com/yungbyun/mllib 의 prediction_util을 이용하여 중고차의 시세를 예측하였습니다.

  --------

  ### 데이터 셋

  해당 캐글에 존재하는 데이터셋을 전처리 하여 이용하였습니다.

  전처리는 https://www.kaggle.com/vnesh123/used-car-price-prediction 을 참고하여 진행하였습니다.

  -------

  1. 모듈 불러오기

   <pre>from mllib.prediction_util import PredictUtil
   import pandas as pd
   </pre>

   예측 모델 객체를 생성하기 위해 mllib 의 PredictUtil 을 prediction_util 에서 import 합니다.

   차후 get_dummies 함수를 사용하기 위해 pandas를 import 합니다.

  2. 객체 생성

   <pre>util = PredictionUtil()
   </pre>

   데이터 셋을 불러오고 예측 알고리즘을 실행하기 위해 객체를 생성합니다.

  3. 데이터 셋 전처리

   <pre>util.read('dataset.csv')
   </pre>

   ![read](https://github.com/kohyerim/ai-project/blob/master/read.png)

   데이터 셋을 read 해오면 다음과 같이 출력됩니다.

   이 때, 필요하지 않은 칼럼들이 보이는데 이를 확인하기 위해 데이터셋 파일을 열어보면 아래와 같은 칼럼들을 확인할 수 있습니다.

   ![dataset](https://github.com/kohyerim/ai-project/blob/master/dataset_pic.png)

   - Unnamed: 0 - 단순한 인덱스 칼럼
   - Location : 중고차가 등록된 주소
   - Kilometers_Driven : 자동차 주행거리
   - Transmission : 변속기 종류
   - Owner_Type : 자동차의 주인이 몇 번째 주인이었는지
   - Seats : 좌석 개수
   - New_Price : 해당 모델의 신차 가격
   - Price : 해당 중고차의 가격
   - Brand_name : 자동차의 제조사 명
   - Mileage_upd : 자동차 회사에서 제공하는 표준 주행 거리
   - Engine_upd : 엔진의 용량 (CC)
   - Power_upd : 엔진의 최대출력 (bhp)
   - Year_upd : 자동차의 제조연도

   위의 칼럼들 중 중고차의 시세와 관계가 없다고 생각되는 칼럼들을 drop합니다.

   <pre>util.drop(['Unnamed: 0', 'brand_name', 'Location', 'New_Price'])
   </pre>

  4. 가격과 그 외의 요소들 간의 상관관계 확인

   - Owner_Type과 가격의 관계 (boxplot)

     <pre>util.boxplot('Owner_Type', 'Price')</pre>

     ![boxplot](https://github.com/kohyerim/ai-project/blob/master/boxplot.png)

     outlier의 값이 많아 가격과의 관계가 크다고 생각되지 않습니다.

   - Power_upd와 가격의 관계 (lmplot)

     <pre>util.lmplot('Power_upd', 'Price', 'Owner_Type')</pre>

     ![lmplot](https://github.com/kohyerim/ai-project/blob/master/lmplot.png)

     완벽한 일차함수의 그래프 모양은 아니지만, 꽤 상관관계가 있어 보입니다.

   - 히트맵 그리기

     - One-Hot-Encoding

       히트맵을 그리기 전에, 숫자가 아닌 칼럼들을 수치화 해주어야 합니다.

       이 때 1, 2, 3 과 같은 대소관계가 있는 수치를 사용하면 안되고 0과 1로 이루어진 새로운 칼럼을 만들어서 수치화 해야 합니다.

       이렇게 수치화 하는 것을 원 핫 인코딩 이라고 합니다.

       <pre>util.df = pd.get_dummies(util.df)</pre>

       Pandas 모듈에서 원 핫 인코딩을 지원하고 있어 객체의 멤버변수인 df에 접근하여 원 핫 인코딩을 해주는 get_dummies 함수를 실행합니다.

       원 핫 인코딩 후의 칼럼은 다음과 같습니다.

       ![get_dummies](https://github.com/kohyerim/ai-project/blob/master/get_dummies.png)

       숫자형이 아닌 칼럼이었던 Fuel_Type, Transmission, Owner_Type 칼럼이 숫자화 된 칼럼으로 변경된 것을 확인할 수 있습니다.

     - 히트맵 출력

       원 핫 인코딩 후, 칼럼들을 모두 이용해 히트맵을 그려보면 다음과 같습니다.

       <pre>util.heatmap(['Kilometers_Driven', 'Seats', 'Price',
                          'Mileage_upd','Engine_upd','Power_upd',
                          'Year_upd', 'Fuel_Type_CNG', 'Fuel_Type_Diesel',
                          'Fuel_Type_Electric', 'Fuel_Type_LPG',
                          'Fuel_Type_Petrol','Transmission_Automatic',
                          'Transmission_Manual', 'Owner_Type_First',
                          'Owner_Type_Fourth & Above', 'Owner_Type_Second',
                          'Owner_Type_Third'])</pre>

       ![heatmap](https://github.com/kohyerim/ai-project/blob/master/Heatmap.png)

       가격(Price)칼럼과 연관성이 높은 칼럼은 'Power_upd', 'Engine_upd', 'Transmission' 정도로 생각이 되어 이를 사용하겠습니다.

       그 외에 'Year_upd', 'Mileage_upd'를 더 이용해 보겠습니다.

  5. 상관관계를 이용하여 중고차 시세 예측하기 - run_all()

   PredictionUtil의 run_all 함수를 호출하면 선형회귀, K-Neighbor, 결정트리, 랜덤포레스트 예측 모델을 실행하여 원하는 값을 예측해 볼 수 있습니다.

   - 'Power_upd', 'Engine_upd', 'Transmission' 칼럼을 이용한 예측

     <pre>util.run_all(['Power_upd', 'Engine_upd', 'Transmission_Manual'], 'Price')</pre>

     'Power_upd', 'Engine_upd', 'Transmission' 칼럼을 이용해 가격을 예측하고, 그 정확도를 확인하면 다음과 같습니다.

     ![run_all](https://github.com/kohyerim/ai-project/blob/master/run_all.png)

     - Linear Regression : 58%
     - K-Neighbor Regression : 74%
     - Decision Tree : 76%
     - Random Forest : 79.9%

   - 'Power_upd', 'Engine_upd', 'Transmission', 'Year_upd' 칼럼을 이용한 예측

     <pre>util.run_all(['Power_upd', 'Engine_upd', 'Transmission_Manual', 'Year_upd'], 'Price')</pre>

     'Power_upd', 'Engine_upd', 'Transmission', 'Year_upd' 칼럼을 이용해 가격을 예측하고, 그 정확도를 확인하면 다음과 같습니다.

     ![addYear](https://github.com/kohyerim/ai-project/blob/master/addYear.png)

     - Linear Regression : 67%
     - K-Neighbor Regression : 82%
     - Decision Tree : 87%
     - Random Forest : 89%

     3개의 칼럼만 이용했을 때 보다 정확도가 더 좋아졌습니다.

   - 'Power_upd', 'Engine_upd', 'Transmission', 'Year_upd', 'Mileage_upd' 칼럼을 이용한 예측

     <pre>util.run_all(['Power_upd', 'Engine_upd', 'Transmission_Manual', 'Year_upd', 'Mileage_upd'], 'Price')</pre>

     'Power_upd', 'Engine_upd', 'Transmission', 'Year_upd', 'Mileage_upd' 칼럼을 이용해 가격을 예측하고, 그 정확도를 확인하면 다음과 같습니다.

     ![addMileage](https://github.com/kohyerim/ai-project/blob/master/addMileage.png)

     - Linear Regression : 67%
     - K-Neighbor Regression : 82%
     - Decision Tree : 88%
     - Random Forest : 89%

     4개의 칼럼을 이용했을 때와 비교하면 정확도가 비슷합니다.
