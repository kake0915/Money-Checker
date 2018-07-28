import cv2
import numpy as np
import random
import collections
import os
import math

#フォント
font = cv2.FONT_HERSHEY_COMPLEX
#いろ
fontColor=(255,255,255)
#サイズ
fontSize=5
#特徴点抽出のための元画像読み込み
coinsDir='images/edge/'
billsDir='images/bills/'
coinsFiles = os.listdir(coinsDir)
billsFiles = os.listdir(billsDir)
#マッチング用オブジェクトの生成
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
#特徴点検出アルゴリズムの決定(AKAZEを使用)
#detector = cv2.AKAZE_create()
detector = cv2.ORB_create()

#画像読み込み
framePath='coins12.jpg'
frame = cv2.imread('images/'+framePath)
#frame = cv2.imread('images/50002.jpg')
img = frame
#入力画像表示
#cv2.namedWindow("inputImage", cv2.WINDOW_NORMAL)
#画像の高さ，幅を取得
height,width,c = img.shape
#グレースケール画像に変換
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#大津の方法により2値化
ret,binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#輪郭検出
binaryImage, contours, hierarchy = cv2.findContours(binaryImage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.namedWindow("Target", cv2.WINDOW_NORMAL)

#合計金額計算用変数
prise = 0
for i,contour in enumerate(contours):
    #輪郭内の面積を計算する
    area = cv2.contourArea(contour)
    #面積が小さいものは何もしない
    if area < 15000:
        continue

    #注目している物体の輪郭点の集合
    cnt = contours[i]
    img = cv2.drawContours(img, [cnt], 0, fontColor, 3)
    #輪郭の上端の座標を取得
    top = tuple(cnt[cnt[:,:,1].argmin()][0])
    #外接矩形の枠の座標取得
    x,y,w,h = cv2.boundingRect(cnt)
    #注目しているターゲットのみを抽出した画像を作成
    target = img[y:y+h,x:x+w,:]
    #画像の高さ，幅を取得
    heightTarget,widthTarget,_= target.shape
    cv2.imshow("Target", target)
    #回転を意識しない外接矩形を描く
    #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #最小外接円の計算
    _,radius = cv2.minEnclosingCircle(cnt)
    #最小外接円の面積
    area2 = radius*radius*(math.pi)
    #ターゲットのキーポイントと特徴記述子を求める
    kp1, des1 = detector.detectAndCompute(target,None)

    #検出した輪郭に囲まれる面積と輪郭の最小外接円の面積の割合が1に近いほど硬貨である
    #紙幣であると分類
    if(area2/area > 1.5):
        minRet=1000
        for file in billsFiles:
            if file == '.DS_Store':
    	        continue
            #見本を読み込み
            imagePath = billsDir+file
            temprate=cv2.imread(imagePath)
            #見本のキーポイントと特徴記述子を求める
            kp2, des2 = detector.detectAndCompute(temprate,None)
            #二つの画像をマッチング
            matches = bf.match(des1, des2)
            dist = [m.distance for m in matches]
    		#類似度を計算する
            ret = sum(dist)/len(dist)
            #類似度が最小なら更新する
            if(ret<minRet):
                minRet=ret
                minFileName=file
        #ターゲットが1000円札なら
        if(minFileName == '1000_1.JPG' or minFileName == '1000_2.JPG'):
            prise +=1000
            cv2.putText(img,'1000yen',top,font, fontSize,fontColor)
        #ターゲットが5000円札なら
        elif(minFileName == '5000_1.JPG' or minFileName == '5000_2.JPG'):
            prise+=5000
            cv2.putText(img,'5000yen',top,font, fontSize,fontColor)
        #ターゲットが10000円札なら
        else:
            prise +=10000
            cv2.putText(img,'10000yen',top,font, fontSize,fontColor)

    #硬貨であると分類
    else:
        #ターゲット画像をhsv空間に変換
        hsvTarget = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        #H空間の平均値計算
        meanHTarget=np.mean(hsvTarget[:,:,0])
        #S空間の平均値計算
        meanSTarget=np.mean(hsvTarget[:,:,1])
        #print(meanHTarget,meanSTarget)
        #ターゲット画像をグレースケール画像に変換
        grayTarget = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        #ターゲット画像の中心から±10の大きさで画像を作成
        h=int(heightTarget/2)
        w=int(widthTarget/2)
        centerTarget = grayTarget[h-10:h+10,w-10:w+10]
        #中央画像の平均値を計算
        meanCenterTarget=np.mean(centerTarget)
        #平均値が50以下の時5か50円玉
        if(meanCenterTarget<50):
            #S空間の平均値が100以上なら5円玉
            if(meanSTarget>90):
                prise+=5
                cv2.putText(img,'5yen',top,font, fontSize,fontColor)
            #そうでないな50円玉
            else:
                prise+=50
                cv2.putText(img,'50yen',top,font, fontSize,fontColor)
        #平均値が50以上でS空間の平均値が100以上かつH空間の平均値が20以下なら10円玉
        elif(meanSTarget>100):
            prise+=10
            cv2.putText(img,'10yen',top,font, fontSize,fontColor)
        #可能性(1,100,500)
        else:
            grayTarget = cv2.GaussianBlur(grayTarget,(5,5),0)
            #エッジ抽出
            edgeTarget = cv2.Canny(grayTarget,50,70)
            #大津の方法により2値化
            ret,edgeTarget = cv2.threshold(edgeTarget, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            #硬貨の形状マッチング
            maxMean=0
            for file in coinsFiles:
                if file == '.DS_Store':
                    continue
                #見本を読み込み
                imagePath = coinsDir+file
                temprate=cv2.imread(imagePath)
                temprate = cv2.cvtColor(temprate, cv2.COLOR_BGR2GRAY)
                #画像表示
                cv2.namedWindow("temprate", cv2.WINDOW_NORMAL)
                cv2.imshow("temprate", temprate)
                #画像の高さ，幅を取得
                heightTemprate,widthTemprate= temprate.shape
                #テンプレートとターゲットの画像の大きさを揃える
                temprate = cv2.resize(temprate, None, fx = widthTarget/widthTemprate, fy = heightTarget/heightTemprate)
                beatAngle=0
                bestMean =0
                for i in range(0,360):
                    # 回転変換行列の算出
                    rotation_matrix = cv2.getRotationMatrix2D((widthTarget/2,heightTarget/2), i, 1.0)
                    # アフィン変換
                    targetRot = cv2.warpAffine(edgeTarget, rotation_matrix, (widthTarget,heightTarget), flags=cv2.INTER_CUBIC)

                    if(bestMean<np.mean(temprate*targetRot)):
                        bestMean=np.mean((temprate*targetRot).flatten())
                        beatAngle=i

                # 回転変換行列の算出
                rotation_matrix = cv2.getRotationMatrix2D((widthTarget/2,heightTarget/2), beatAngle, 1.0)
                # アフィン変換
                targetRot = cv2.warpAffine(edgeTarget, rotation_matrix, (widthTarget,heightTarget), flags=cv2.INTER_CUBIC)
                cv2.waitKey(0)
                cv2.imshow("Target", temprate*targetRot)
                cv2.waitKey(0)
                if(maxMean<bestMean):
                    maxMean=bestMean
                    maxFileName = file

            print(maxFileName)
            #ターゲットが100円玉なら
            if(maxFileName == '100_1.JPG' or maxFileName == '100_2.JPG'):
                prise +=100
                cv2.putText(img,'100yen',top,font, fontSize,fontColor)
            #ターゲットが1円玉なら
            elif(maxFileName == '1_1.JPG' or maxFileName == '1_2.JPG'):
                prise+=1
                cv2.putText(img,'1yen',top,font, fontSize,fontColor)
            #ターゲットが500円玉なら
            else:
                prise +=500
                cv2.putText(img,'500yen',top,font, fontSize,fontColor)

    cv2.waitKey(0)
outPrise=str(prise)+'yen'
print(outPrise)
cv2.putText(img,outPrise,(100,300),font, fontSize+5,fontColor)
#画面出力
cv2.namedWindow("outputImage", cv2.WINDOW_NORMAL)
cv2.imshow("outputImage", img)
cv2.imwrite('images/imageOut/'+framePath, img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#回転を考慮した外接矩形を描く
#外接矩形の計算(戻り値：左上の点(x,y)，横と縦のサイズ(width, height)，回転角)
rect = cv2.minAreaRect(cnt)
rectから外接矩形の四隅の点を計算
box = cv2.boxPoints(rect)
box = np.int0(box)
#外接矩形を描く
img = cv2.drawContours(img,[box],0,(0,0,255),2)

# 最初の（上位）10個の対応点を描画
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(target,kp1,temprate,kp2,matches[:10, None,flags=2)
cv2.namedWindow("outputImage2", cv2.WINDOW_NORMAL)
cv2.imshow("outputImage2",img3 )

#硬貨の特徴点比較
minRet=1000
for file in coinsFiles:
    if file == '.DS_Store':
        continue
    #見本を読み込み
    imagePath = coinsDir+file
    temprate=cv2.imread(imagePath)
    #見本のキーポイントと特徴記述子を求める
    kp2, des2 = detector.detectAndCompute(temprate,None)
    #二つの画像をマッチング
    matches = bf.match(des1, des2)
    dist = [m.distance for m in matches]
    #類似度を計算する
    ret = sum(dist)/len(dist)
    #類似度が最小なら更新する
    if(ret<minRet):
        minRet=ret
        minFileName=file


# カーネルの定義
kernel = np.ones((6, 6), np.uint8)
# 膨張・収縮処理
grayTarget = cv2.dilate(grayTarget, kernel)
grayTarget = cv2.erode(grayTarget, kernel)
#エッジ抽出
edgeTarget = cv2.Canny(grayTarget,50,70)
# 膨張・収縮処理
#edgeTarget = cv2.dilate(edgeTarget, kernel)
#edgeTarget = cv2.erode(edgeTarget, kernel)
#大津の方法により2値化
_,binaryTarget = cv2.threshold(grayTarget, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
edgeTarget*=binaryTarget
#画面出力
cv2.imshow("Target", edgeTarget)

#ターゲットのキーポイントと特徴記述子を求める
kp3, des3 = detector.detectAndCompute(target,None)
#硬貨の特徴点比較
minRet=1000
for file in coinsFiles:
    if file == '.DS_Store':
        continue
    #見本を読み込み
    imagePath = coinsDir+file
    temprate=cv2.imread(imagePath)
    #見本のキーポイントと特徴記述子を求める
    kp4, des4 = detector.detectAndCompute(temprate,None)
    #二つの画像をマッチング
    matches = bf.match(des3, des4)
    dist = [m.distance for m in matches]
    #類似度を計算する
    ret = sum(dist)/len(dist)
    #類似度が最小なら更新する
    if(ret<minRet):
        minRet=ret
        minFileName=file

    # 最初の（上位）10個の対応点を描画
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(target,kp3,temprate,kp4,matches[:10], None,flags=2)
    cv2.namedWindow("outputImage2", cv2.WINDOW_NORMAL)
    cv2.imshow("outputImage2",img3 )
    cv2.waitKey(0)
'''
