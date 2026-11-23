import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
	id: window
	width: 1500
	height: 920
	minimumWidth: 1000
	minimumHeight: 600
	visible: true
	title: app.title ? app.title + " — note-OCR" : "note-OCR"
	color: theme.background

	palette.window: theme.panel
	palette.windowText: theme.text
	palette.base: theme.panelRaised
	palette.text: theme.text
	palette.button: theme.panelRaised
	palette.buttonText: theme.text
	palette.highlight: theme.accent
	palette.highlightedText: "#101722"
	palette.mid: theme.border

	property real zoom: 1.5

	Connections {
		target: app
		function onMessage(text) { toast.show(text) }
		function onFocusMeasure(x, y, w, h) {
			flick.contentX = Math.max(0, (x + w / 2) * window.zoom - flick.width / 2)
			flick.contentY = Math.max(0, (y + h / 2) * window.zoom - flick.height / 2)
			measureFlash.x = x * window.zoom; measureFlash.y = y * window.zoom
			measureFlash.width = w * window.zoom; measureFlash.height = h * window.zoom
			flashAnim.restart()
		}
		function onPageChanged() { fitWidth() }
	}

	function fitWidth() {
		if (app.pageWidth > 0) window.zoom = Math.max(0.3, (flick.width - 24) / app.pageWidth)
	}

	// ------------------------------------------------------------- dialogs
	FileDialog {
		id: openDialog
		title: "Open sheet music"
		nameFilters: ["Sheet music (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf)", "All files (*)"]
		onAccepted: app.openFile(selectedFile)
	}
	FileDialog {
		id: xmlDialog
		fileMode: FileDialog.SaveFile
		defaultSuffix: "musicxml"
		nameFilters: ["MusicXML (*.musicxml)"]
		onAccepted: app.exportXml(selectedFile)
	}
	FileDialog {
		id: msczDialog
		fileMode: FileDialog.SaveFile
		defaultSuffix: "mscz"
		nameFilters: ["MuseScore (*.mscz)"]
		onAccepted: app.exportMscz(selectedFile)
	}
	FileDialog {
		id: appendDialog
		title: "Append to which file?"
		nameFilters: ["MuseScore or MusicXML (*.mscz *.musicxml *.xml)"]
		onAccepted: app.appendTo(selectedFile, selectedFile)
	}

	// ------------------------------------------------------------- shortcuts
	Shortcut { sequences: [StandardKey.Open]; onActivated: openDialog.open() }
	Shortcut { sequence: "Ctrl+E"; onActivated: xmlDialog.open() }
	Shortcut { sequence: "Ctrl+Shift+E"; onActivated: msczDialog.open() }
	Shortcut { sequence: "Space"; onActivated: app.togglePlay() }
	Shortcut { sequence: "Up"; onActivated: app.step(1) }  // pitch one step
	Shortcut { sequence: "Down"; onActivated: app.step(-1) }
	Shortcut { sequence: "#"; onActivated: app.accidental(1) }
	Shortcut { sequence: "B"; onActivated: app.accidental(-1) }
	Shortcut { sequence: "Shift+N"; onActivated: app.accidental(0) }
	Shortcut { sequence: "1"; onActivated: app.duration("4") }
	Shortcut { sequence: "2"; onActivated: app.duration("2") }
	Shortcut { sequence: "3"; onActivated: app.duration("1") }
	Shortcut { sequence: "4"; onActivated: app.duration("1/2") }
	Shortcut { sequence: "5"; onActivated: app.duration("1/4") }
	Shortcut { sequence: "6"; onActivated: app.duration("1/8") }
	Shortcut { sequence: "."; onActivated: app.toggleDot() }
	Shortcut { sequence: "Delete"; onActivated: app.deleteSelected() }
	Shortcut { sequence: "A"; onActivated: app.setTool(app.tool === "add" ? "select" : "add") }
	Shortcut { sequence: "Escape"; onActivated: { app.setTool("select"); app.select(-1) } }
	Shortcut { sequence: "N"; onActivated: app.nextBad(1) }
	Shortcut { sequence: "P"; onActivated: app.nextBad(-1) }
	Shortcut { sequences: ["+", "="]; onActivated: window.zoom = Math.min(6, window.zoom * 1.25) }
	Shortcut { sequence: "-"; onActivated: window.zoom = Math.max(0.2, window.zoom / 1.25) }
	Shortcut { sequence: "0"; onActivated: fitWidth() }

	component ToolBtn: Button {
		flat: true
		padding: 8
		font.pixelSize: 13
	}
	component SectionLabel: Label {
		color: theme.faint
		font.pixelSize: 11
		font.capitalization: Font.AllUppercase
		font.letterSpacing: 1
	}

	ColumnLayout {
		anchors.fill: parent
		spacing: 0

		// ------------------------------------------------------------- toolbar
		Rectangle {
			Layout.fillWidth: true
			Layout.preferredHeight: 48
			color: theme.panel
			Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: theme.border }
			RowLayout {
				anchors.fill: parent
				anchors.leftMargin: 10
				anchors.rightMargin: 10
				spacing: 4
				ToolBtn { text: "Open…"; onClicked: openDialog.open() }
				Rectangle { width: 1; height: 24; color: theme.border }
				ToolBtn {
					text: app.playing ? "■ Stop" : "▶ Play"
					enabled: app.pageCount > 0
					onClicked: app.togglePlay()
				}
				Rectangle { width: 1; height: 24; color: theme.border }
				ToolBtn { text: "Export MusicXML"; enabled: app.pageCount > 0; onClicked: xmlDialog.open() }
				ToolBtn { text: "Export .mscz"; enabled: app.pageCount > 0; onClicked: msczDialog.open() }
				ToolBtn { text: "Append to…"; enabled: app.pageCount > 0; onClicked: appendDialog.open() }
				Item { Layout.fillWidth: true }
				Label {
					visible: app.busy !== ""
					text: app.busy + "…"
					color: theme.warn
				}
				BusyIndicator { visible: app.busy !== ""; running: visible; implicitWidth: 28; implicitHeight: 28 }
				Rectangle { width: 1; height: 24; color: theme.border }
				ToolBtn { text: "−"; onClicked: window.zoom = Math.max(0.2, window.zoom / 1.25) }
				Label { text: Math.round(window.zoom * 100) + " %"; color: theme.muted; Layout.preferredWidth: 48; horizontalAlignment: Text.AlignHCenter }
				ToolBtn { text: "+"; onClicked: window.zoom = Math.min(6, window.zoom * 1.25) }
				ToolBtn { text: "Fit"; onClicked: fitWidth() }
			}
		}

		SplitView {
			Layout.fillWidth: true
			Layout.fillHeight: true
			orientation: Qt.Horizontal

			// ------------------------------------------------------------- sidebar
			Rectangle {
				SplitView.preferredWidth: 230
				SplitView.minimumWidth: 170
				color: theme.panel
				ColumnLayout {
					anchors.fill: parent
					anchors.margins: 10
					spacing: 6
					SectionLabel { text: "Pages" }
					Repeater {
						model: app.pageCount
						delegate: ItemDelegate {
							required property int index
							Layout.fillWidth: true
							text: "Page " + (index + 1)
							highlighted: index === app.currentPage
							onClicked: app.setPage(index)
						}
					}
					SectionLabel { text: "Measures"; Layout.topMargin: 8 }
					ListView {
						id: measureList
						Layout.fillWidth: true
						Layout.fillHeight: true
						clip: true
						model: app.measures
						ScrollBar.vertical: ScrollBar {}
						delegate: ItemDelegate {
							required property var modelData
							width: measureList.width
							height: 28
							contentItem: RowLayout {
								spacing: 6
								Rectangle {
									width: 8; height: 8; radius: 4
									color: modelData.ok ? theme.ok : theme.error
								}
								Label {
									Layout.fillWidth: true
									text: modelData.label
									color: modelData.ok ? theme.text : theme.error
									elide: Text.ElideRight
								}
								Label { text: "p" + (modelData.page + 1); color: theme.faint; font.pixelSize: 11 }
							}
							onClicked: app.gotoMeasure(modelData.index)
						}
					}
					Label {
						Layout.fillWidth: true
						wrapMode: Text.WordWrap
						color: theme.faint
						font.pixelSize: 11
						text: "N / P: next / previous measure to check"
					}
				}
			}

			// ------------------------------------------------------------- page view
			Rectangle {
				SplitView.fillWidth: true
				color: theme.background
				clip: true

				Label {
					anchors.centerIn: parent
					visible: app.pageCount === 0
					text: app.busy !== "" ? app.busy + "…" : "Open an image or PDF of sheet music  (Ctrl+O)"
					color: theme.muted
					font.pixelSize: 16
				}

				Flickable {
					id: flick
					anchors.fill: parent
					contentWidth: pageItem.width + 24
					contentHeight: pageItem.height + 24
					boundsBehavior: Flickable.StopAtBounds
					ScrollBar.vertical: ScrollBar {}
					ScrollBar.horizontal: ScrollBar {}
					onWidthChanged: if (app.pageCount > 0 && window.zoom < 0.31) fitWidth()

					Item {
						id: pageItem
						x: 12
						y: 12
						width: app.pageWidth * window.zoom
						height: app.pageHeight * window.zoom
						visible: app.pageCount > 0

						Rectangle { anchors.fill: parent; anchors.margins: -1; color: "white" }
						Image {
							anchors.fill: parent
							source: app.pageUrl
							smooth: true
							mipmap: true
							fillMode: Image.Stretch
						}
						// measures that don't add up
						Repeater {
							model: app.measures
							delegate: Rectangle {
								required property var modelData
								visible: modelData.page === app.currentPage && !modelData.ok
								x: modelData.x * window.zoom
								y: modelData.y * window.zoom
								width: modelData.w * window.zoom
								height: modelData.h * window.zoom
								color: "#22f28b82"
								border.color: "#88f28b82"
								border.width: 1
							}
						}
						Rectangle {
							id: measureFlash
							color: "transparent"
							border.color: theme.accent
							border.width: 3
							opacity: 0
							SequentialAnimation on opacity {
								id: flashAnim
								running: false
								NumberAnimation { to: 1; duration: 120 }
								PauseAnimation { duration: 600 }
								NumberAnimation { to: 0; duration: 500 }
							}
						}
						// symbols
						Repeater {
							model: app.symbols
							delegate: Rectangle {
								required property var modelData
								readonly property bool sel: modelData.uid === app.selected
								readonly property bool playing: modelData.uid === app.playUid
								x: modelData.x * window.zoom - 1
								y: modelData.y * window.zoom - 1
								width: Math.max(2, modelData.w * window.zoom + 2)
								height: Math.max(2, modelData.h * window.zoom + 2)
								color: playing ? "#5572a7ff" : (sel ? "#3372a7ff" : "transparent")
								border.width: sel || playing ? 2 : 1
								border.color: sel ? "#ffffff" : modelData.color
								opacity: sel || playing ? 1 : (modelData.score < 0.5 ? 0.9 : 0.45)
							}
						}
						MouseArea {
							anchors.fill: parent
							cursorShape: app.tool === "add" ? Qt.CrossCursor : Qt.ArrowCursor
							onClicked: (m) => app.clickAt(m.x / window.zoom, m.y / window.zoom)
							onWheel: (w) => {
								if (w.modifiers & Qt.ControlModifier) {
									window.zoom = Math.max(0.2, Math.min(6, window.zoom * (w.angleDelta.y > 0 ? 1.15 : 1 / 1.15)))
								} else {
									w.accepted = false
								}
							}
						}
					}
				}
			}

			// ------------------------------------------------------------- inspector
			Rectangle {
				SplitView.preferredWidth: 380
				SplitView.minimumWidth: 300
				color: theme.panel
				ColumnLayout {
					anchors.fill: parent
					spacing: 0
					TabBar {
						id: tabs
						objectName: "tabs"
						Layout.fillWidth: true
						TabButton { text: "Note" }
						TabButton { text: "Result" }
					}
					StackLayout {
						Layout.fillWidth: true
						Layout.fillHeight: true
						currentIndex: tabs.currentIndex

						// --- Note / symbol inspector
						ScrollView {
							id: inspScroll
							clip: true
							contentWidth: availableWidth
							ColumnLayout {
								width: inspScroll.availableWidth
								spacing: 8
								Item { height: 4 }
								Label {
									Layout.leftMargin: 12
									visible: !app.selInfo.cls
									text: "Click a symbol on the page to edit it.\nA: add tool · Del: delete · ↑↓: pitch\n1–6: duration · . dot · # B ⇧N: accidentals\nSpace: play"
									color: theme.muted
									lineHeight: 1.3
								}
								ColumnLayout {
									visible: !!app.selInfo.hasEvent
									Layout.leftMargin: 12
									Layout.rightMargin: 12
									spacing: 6
									SectionLabel { text: "Note" }
									Label { text: app.selInfo.note || ""; font.pixelSize: 22; color: theme.text }
									Label { text: app.selInfo.duration || ""; color: theme.muted }
									Label { visible: !!app.selInfo.marks; text: app.selInfo.marks || ""; color: theme.accent }
									Flow {
										Layout.fillWidth: true
										spacing: 4
										Button { text: "↑"; onClicked: app.step(1) }
										Button { text: "↓"; onClicked: app.step(-1) }
										Button { text: "♯"; onClicked: app.accidental(1) }
										Button { text: "♭"; onClicked: app.accidental(-1) }
										Button { text: "♮"; onClicked: app.accidental(0) }
									}
									Flow {
										Layout.fillWidth: true
										spacing: 4
										Repeater {
											model: [["𝅝", "4"], ["𝅗𝅥", "2"], ["♩", "1"], ["♪", "1/2"], ["16", "1/4"], ["32", "1/8"]]
											delegate: Button {
												required property var modelData
												text: modelData[0]
												implicitWidth: 44
												onClicked: app.duration(modelData[1])
											}
										}
										Button { text: "dot"; onClicked: app.toggleDot() }
										Button { text: "→ rest"; onClicked: app.toggleRest() }
									}
								}
								ColumnLayout {
									visible: !!app.selInfo.cls
									Layout.leftMargin: 12
									Layout.rightMargin: 12
									Layout.topMargin: 8
									spacing: 6
									SectionLabel { text: "Symbol (training label)" }
									RowLayout {
										ComboBox {
											id: clsBox
											Layout.fillWidth: true
											model: app.classes
											currentIndex: app.classes.indexOf(app.selInfo.cls || "")
											onActivated: app.setClass(currentText)
										}
										Button { text: "Delete"; onClicked: app.deleteSelected() }
									}
									Label { text: "confidence " + (app.selInfo.score ?? ""); color: theme.faint; font.pixelSize: 11 }
								}
								ColumnLayout {
									Layout.leftMargin: 12
									Layout.rightMargin: 12
									Layout.topMargin: 12
									spacing: 6
									SectionLabel { text: "Add missing symbol" }
									RowLayout {
										ComboBox {
											Layout.fillWidth: true
											model: app.classes
											currentIndex: app.classes.indexOf(app.addClass)
											onActivated: app.setAddClass(currentText)
										}
										Button {
											text: app.tool === "add" ? "Adding… (Esc)" : "Add (A)"
											highlighted: app.tool === "add"
											onClicked: app.setTool(app.tool === "add" ? "select" : "add")
										}
									}
									Label {
										Layout.fillWidth: true
										wrapMode: Text.WordWrap
										color: theme.faint
										font.pixelSize: 11
										text: "Symbol fixes become training data. When you have a few pages fixed, retrain the model with them."
									}
									Button {
										text: "Retrain with my corrections"
										enabled: app.busy === ""
										onClicked: app.retrain()
									}
								}
							}
						}

						// --- MuseScore render of the result
						Item {
							Label {
								anchors.centerIn: parent
								visible: app.resultImages.length === 0
								text: app.resultBusy ? "Rendering with MuseScore…" : "The recognised music appears here"
								color: theme.muted
							}
							ScrollView {
								id: resultScroll
								anchors.fill: parent
								clip: true
								contentWidth: availableWidth
								ColumnLayout {
									width: resultScroll.availableWidth
									spacing: 8
									Repeater {
										model: app.resultImages
										delegate: Image {
											required property var modelData
											Layout.fillWidth: true
											Layout.preferredHeight: width * (implicitHeight / Math.max(1, implicitWidth))
											source: modelData
											fillMode: Image.PreserveAspectFit
											smooth: true
											mipmap: true
											Rectangle { anchors.fill: parent; z: -1; color: "white" }
										}
									}
								}
							}
							Button {
								anchors.right: parent.right
								anchors.bottom: parent.bottom
								anchors.margins: 10
								text: app.resultBusy ? "Rendering…" : "Refresh"
								enabled: !app.resultBusy
								onClicked: app.renderResult()
							}
						}
					}
				}
			}
		}

		// ------------------------------------------------------------- status bar
		Rectangle {
			Layout.fillWidth: true
			Layout.preferredHeight: 28
			color: theme.panel
			Rectangle { anchors.top: parent.top; width: parent.width; height: 1; color: theme.border }
			Label {
				anchors.verticalCenter: parent.verticalCenter
				anchors.left: parent.left
				anchors.leftMargin: 12
				text: app.summary
				color: theme.muted
				font.pixelSize: 12
			}
		}
	}

	// ------------------------------------------------------------- toast
	Rectangle {
		id: toast
		property alias text: toastLabel.text
		function show(t) { text = t; opacity = 1; toastTimer.restart() }
		anchors.horizontalCenter: parent.horizontalCenter
		anchors.bottom: parent.bottom
		anchors.bottomMargin: 48
		width: toastLabel.implicitWidth + 32
		height: 36
		radius: 8
		color: theme.panelRaised
		border.color: theme.border
		opacity: 0
		Behavior on opacity { NumberAnimation { duration: 200 } }
		Label { id: toastLabel; anchors.centerIn: parent; color: theme.text }
		Timer { id: toastTimer; interval: 3500; onTriggered: toast.opacity = 0 }
	}
}
